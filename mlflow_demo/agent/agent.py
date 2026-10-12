"""THE banking customer-support agent — one implementation shared by the
workshop notebooks AND the Databricks app (the app's former telco agent).

Built on the OpenAI Agents SDK (`Agent` + `Runner`), calling the participant's
prefixed UC banking functions as tools via DatabricksFunctionClient (serverless
execution), and routed through the participant's AI-Gateway model service
(<prefix>_bank_agent_llm) when LLM_MODEL_SERVICE is set — falling back to the
plain LLM_MODEL serving endpoint otherwise.

MLflow tracing: autolog is registered at module import (the patched OpenAI
client is the one the Agents SDK Runner uses), @mlflow.trace wraps the sync
entry points so a whole agent run nests under one AGENT-typed trace with a
TOOL-typed child span per UC function call.

Env contract (set by setup_databricks_notebook_env() in notebooks and by the
DAB app config in the app): UC_CATALOG, UC_SCHEMA, PROMPT_NAME, PROMPT_ALIAS,
UC_TOOL_SCHEMA, UC_TOOL_PREFIX, LLM_MODEL_SERVICE, LLM_MODEL,
DEMO_CUSTOMER_ID (default CUST-00015 — the customer context for notebook
questions phrased as "my account", mirroring the app UI's customer selector).
"""

import asyncio
import contextvars
import logging
import os
import threading
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Optional

import mlflow
from agents import (
  Agent,
  Runner,
  function_tool,
  set_default_openai_api,
  set_default_openai_client,
)
from agents.tracing import set_trace_processors
from databricks_openai import AsyncDatabricksOpenAI, DatabricksFunctionClient
from mlflow.entities import SpanType
from mlflow.types.responses import (
  Message,
  OutputItem,
  ResponsesAgentRequest,
  ResponsesAgentResponse,
  ResponsesAgentStreamEvent,
  create_text_output_item,
)

from .prompts import ORIGINAL_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

UC_CATALOG = os.getenv('UC_CATALOG', '')
UC_SCHEMA = os.getenv('UC_SCHEMA', '')
PROMPT_NAME = os.getenv('PROMPT_NAME', '')
PROMPT_ALIAS = os.getenv('PROMPT_ALIAS', 'production')
LLM_MODEL_SERVICE = os.getenv('LLM_MODEL_SERVICE', '')
LLM_MODEL = os.getenv('LLM_MODEL', 'databricks-claude-sonnet-5')
DEMO_CUSTOMER_ID = os.getenv('DEMO_CUSTOMER_ID', 'CUST-00015')

# The model the Agents SDK Runner calls: the governed AI-Gateway model service
# when configured, else the plain serving endpoint.
MODEL = LLM_MODEL_SERVICE or LLM_MODEL

# API + tracing + OpenAI client setup at module import — the proven app
# pattern: ONE client created once (the Agents SDK Runner, autolog, and the
# http connection pool all bind to it; re-creating WorkspaceClients per
# request inside the app runtime intermittently fails auth on the 2nd+
# request). Both entry points drain the Runner inside asyncio.run and this
# single client has served multiple sequential requests per process
# correctly (httpx re-binds its connections to each run's event loop).
# Route through the participant's AI-Gateway model service when configured —
# via the OpenAI-NATIVE gateway path ({host}/ai-gateway/openai/v1,
# use_ai_gateway_native_api=True), which returns proper OpenAI-shaped deltas
# (the mlflow/v1 path passes Anthropic content blocks through, which the
# Agents SDK's strict event models reject) — else the plain serving endpoint.
# ---------------------------------------------------------------------------
# Reasoning-block normalization. Claude reasoning models (Sonnet 5)
# stochastically emit Anthropic-style CONTENT BLOCKS (e.g. {"type":
# "reasoning", ...}) inside the OpenAI-compat streaming delta.content — as a
# LIST where OpenAI clients expect a string. The OpenAI Agents SDK's strict
# event models reject that ("ResponseTextDeltaEvent ... Input should be a
# valid string"). Wrap the client's chat.completions to flatten any
# block-list content: keep text blocks' text, drop reasoning/redacted blocks.
# ---------------------------------------------------------------------------
from databricks_openai.utils.clients import (
  AsyncDatabricksChat,
  AsyncDatabricksCompletions,
)
from openai.resources.chat.chat import AsyncChat


def _flatten_content(content) -> Any:
  """Anthropic content-block list -> concatenated text ('' if no text)."""
  if not isinstance(content, list):
    return content
  parts = []
  for block in content:
    if isinstance(block, dict):
      if block.get('type') in ('text', 'output_text', 'input_text'):
        parts.append(block.get('text') or '')
    elif isinstance(block, str):
      parts.append(block)
  return ''.join(parts)


class _ReasoningSafeAsyncStream:
  """AsyncStream wrapper that flattens block-list delta.content per chunk."""

  def __init__(self, stream):
    self._stream = stream

  def __aiter__(self):
    return self

  async def __anext__(self):
    chunk = await self._stream.__anext__()
    for choice in getattr(chunk, 'choices', None) or []:
      delta = getattr(choice, 'delta', None)
      content = getattr(delta, 'content', None)
      if delta is not None and isinstance(content, list):
        try:
          object.__setattr__(delta, 'content', _flatten_content(content))
        except Exception:
          pass
    return chunk

  async def close(self):
    await self._stream.close()

  def __getattr__(self, name):
    return getattr(self._stream, name)


class _ReasoningSafeCompletions(AsyncDatabricksCompletions):
  """chat.completions.create with content-block normalization on top of the
  databricks_openai Claude fixes (strict-stripping, empty-content)."""

  async def create(self, **kwargs):
    result = await super().create(**kwargs)
    if kwargs.get('stream'):
      return _ReasoningSafeAsyncStream(result)
    for choice in getattr(result, 'choices', None) or []:
      msg = getattr(choice, 'message', None)
      if msg is not None and isinstance(getattr(msg, 'content', None), list):
        try:
          object.__setattr__(msg, 'content', _flatten_content(msg.content))
        except Exception:
          pass
    return result


class _ReasoningSafeChat(AsyncChat):
  def __init__(self, client):
    super().__init__(client=client)
    self.completions = _ReasoningSafeCompletions(client=client)


class _GatewayOpenAI(AsyncDatabricksOpenAI):
  """AsyncDatabricksOpenAI whose chat.completions normalizes reasoning blocks
  (cached per instance — the base .chat property rebuilds on every access)."""

  @property
  def chat(self) -> Any:
    cached = getattr(self, '_rs_chat', None)
    if cached is None:
      base_chat = super().chat
      if isinstance(base_chat, AsyncDatabricksChat):
        base_chat.completions = _ReasoningSafeCompletions(
            client=base_chat.completions._client
        )
        cached = base_chat
      else:
        cached = _ReasoningSafeChat(client=base_chat.completions._client)
      object.__setattr__(self, '_rs_chat', cached)
    return cached


def _make_client() -> AsyncDatabricksOpenAI:
  """Build the OpenAI client: the participant's AI-Gateway model service via
  the OpenAI-NATIVE path ({host}/ai-gateway/openai/v1) when LLM_MODEL_SERVICE
  is set, else the plain Databricks serving endpoint — always wrapped for
  reasoning-block safety."""
  if LLM_MODEL_SERVICE:
    return _GatewayOpenAI(use_ai_gateway_native_api=True)
  return _GatewayOpenAI()


# ---------------------------------------------------------------------------
# API + tracing + OpenAI client setup at module import — the proven app
# pattern: ONE client created once (the Agents SDK Runner, autolog, and the
# http connection pool all bind to it; re-creating WorkspaceClients per
# request inside the app runtime intermittently fails auth on the 2nd+
# request). Both entry points drain the Runner inside asyncio.run and this
# single client has served multiple sequential requests per process
# correctly (httpx re-binds its connections to each run's event loop).
# ---------------------------------------------------------------------------
set_default_openai_client(_make_client())
set_default_openai_api('chat_completions')
set_trace_processors([])  # let MLflow be the only trace processor
mlflow.openai.autolog()
logging.getLogger('mlflow.utils.autologging_utils').setLevel(logging.ERROR)


def _configure_client() -> None:
  """Set a fresh OpenAI client on the Agents SDK, bound to the CURRENT event
  loop (for short-lived asyncio.run loops, e.g. notebook predict() calls).
  The app path never calls this — it reuses the module-level client on the
  FastAPI process's one persistent event loop instead."""
  set_default_openai_client(_make_client())


# ---------------------------------------------------------------------------
# Prompt resolution. The default SYSTEM_PROMPT loads the UC-registry prompt
# FRESH ON EACH CALL (so notebook 3's A/B tests, notebook 6's GEPA candidates,
# and the app all serve whatever the registry currently holds), falling back
# to the built-in template when the registry prompt doesn't exist. Notebook 3
# hot-swaps the module global (`agent_module.SYSTEM_PROMPT = load_prompt(...)`),
# which simply takes precedence from then on.
# ---------------------------------------------------------------------------
class _RegistryPrompt:
  """Default SYSTEM_PROMPT: resolves the registry prompt per call, with the
  built-in template as fallback. Duck-types MLflow's Prompt (.format())."""

  @staticmethod
  def format(**kwargs) -> str:
    if UC_CATALOG and UC_SCHEMA and PROMPT_NAME:
      try:
        prompt = mlflow.genai.load_prompt(
            f'prompts:/{UC_CATALOG}.{UC_SCHEMA}.{PROMPT_NAME}@{PROMPT_ALIAS}'
        )
        return prompt.format(**kwargs)
      except Exception as e:
        # Not registered yet — REGISTER it as v1 (this is the notebook-1
        # contract: the first user-context import registers the base prompt;
        # the app's SP cannot CREATE and just falls back until then).
        from ..utils.mlflow_helpers import load_or_register_prompt
        text, source = load_or_register_prompt(
            UC_CATALOG, UC_SCHEMA, PROMPT_NAME,
            ORIGINAL_PROMPT_TEMPLATE, register_if_missing=True,
        )
        if source in ('registered', 'registry'):
          return text
        logger.info('Registry prompt %s unavailable (%s); using built-in template.',
                    PROMPT_NAME, e)
    return ORIGINAL_PROMPT_TEMPLATE


SYSTEM_PROMPT = _RegistryPrompt()


def get_system_prompt_text() -> str:
  """The current system prompt text (module-global override, else registry)."""
  return SYSTEM_PROMPT.format()


# ---------------------------------------------------------------------------
# Tools — the 7 banking UC functions, resolved from the env contract
# (<catalog>.<schema>.<prefix>_get_*). Names and docstrings drive the LLM's
# tool selection. Serverless execution: no SQL warehouse grant needed, the
# caller only needs USE SCHEMA + EXECUTE.
# ---------------------------------------------------------------------------
TARGET_FN_PREFIX = os.getenv('UC_TOOL_SCHEMA') or ''
_FN_NAME_PREFIX = (lambda x: f'{x}_' if x else '')(os.getenv('UC_TOOL_PREFIX', ''))

_client: Optional[DatabricksFunctionClient] = None


def _get_client() -> DatabricksFunctionClient:
  global _client
  if _client is None:
    profile = os.getenv('DATABRICKS_CONFIG_PROFILE')
    kwargs = {'execution_mode': 'serverless'}
    if profile:
      kwargs['profile'] = profile
    _client = DatabricksFunctionClient(**kwargs)
  return _client


@mlflow.trace(span_type=SpanType.TOOL)
def _execute_sync(fn_name: str, params: dict) -> str:
  """Run the UC function on serverless and return its first-column value.

  Traced as a TOOL span so each tool call shows as a child span under the
  agent trace (the notebooks teach exactly this structure).
  """
  client = _get_client()
  result = client.execute_function(
      f'{TARGET_FN_PREFIX}.{_FN_NAME_PREFIX}{fn_name}', params
  )
  if getattr(result, 'error', None):
    return f'{{"error": "{result.error}"}}'
  value = getattr(result, 'value', None)
  return value or '{}'


async def _call_function(fn_name: str, params: dict) -> str:
  """Async wrapper that executes the UC function in a worker thread."""
  return await asyncio.to_thread(_execute_sync, fn_name, params)


@function_tool
async def get_customer_profile(customer_id: str) -> str:
  """Retrieve a customer's profile: name, email, phone, tier (Standard/Premium/
  Private Banking), segment, and customer-since date.

  Args:
    customer_id: Customer ID like 'CUST-00001'.
  """
  return await _call_function('get_customer_profile', {'p_customer_id': customer_id})


@function_tool
async def get_account_balance(customer_id: str) -> str:
  """List all of a customer's active accounts with their type (Checking/Savings/
  CD), balance, and currency.

  Args:
    customer_id: Customer ID like 'CUST-00001'.
  """
  return await _call_function('get_account_balance', {'p_customer_id': customer_id})


@function_tool
async def get_transaction_history(customer_id: str, days_back: Optional[int] = None) -> str:
  """Retrieve a customer's transactions (date, amount, merchant, category) for
  the last N days.

  Args:
    customer_id: Customer ID like 'CUST-00001'.
    days_back: Number of days to look back (e.g., 30). Defaults to 90.
  """
  return await _call_function(
      'get_transaction_history',
      {'p_customer_id': customer_id, 'p_days_back': days_back or 90},
  )


@function_tool
async def get_loan_details(customer_id: str) -> str:
  """List a customer's loans with type (Mortgage/Auto/Personal), principal,
  interest rate, term, outstanding balance, and monthly payment.

  Args:
    customer_id: Customer ID like 'CUST-00001'.
  """
  return await _call_function('get_loan_details', {'p_customer_id': customer_id})


@function_tool
async def get_credit_card_info(customer_id: str) -> str:
  """List a customer's active credit cards with type, credit limit, current
  balance, and available credit.

  Args:
    customer_id: Customer ID like 'CUST-00001'.
  """
  return await _call_function('get_credit_card_info', {'p_customer_id': customer_id})


@function_tool
async def get_product_catalog(category: Optional[str] = None) -> str:
  """List banking products, optionally filtered by category.

  Args:
    category: Product category: Account, Loan, Card, or Investment. Omit for all products.
  """
  return await _call_function('get_product_catalog', {'p_category': category})


@function_tool
async def get_branch_info(city: Optional[str] = None) -> str:
  """List bank branches, optionally filtered by city (e.g., Springfield, Fairview,
  Riverton). Returns address, phone, hours, and services.

  Args:
    city: City name to filter by. Omit for all branches.
  """
  return await _call_function('get_branch_info', {'p_city': city})


TOOLS = [
  get_customer_profile,
  get_account_balance,
  get_transaction_history,
  get_loan_details,
  get_credit_card_info,
  get_product_catalog,
  get_branch_info,
]

# Backward-compatible alias for the app layer.
TELCO_TOOLS = TOOLS


# ---------------------------------------------------------------------------
# Agent construction + conversation shaping
# ---------------------------------------------------------------------------
class TextOutputItem(OutputItem):
  """OutputItem whose content parts always expose `.text` — keeps the
  notebooks' duck-parsing (`item.content` -> parts with `.text`) working
  across mlflow versions that store the parts as plain dicts."""

  def model_post_init(self, __context) -> None:
    parts = []
    for p in self.content or []:
      parts.append(p if hasattr(p, 'text') else SimpleNamespace(text=p.get('text', '')))
    object.__setattr__(self, 'content', parts)


def build_agent(model: Optional[str] = None) -> Agent:
  """Build the banking support Agent with all UC function tools.

  The prompt resolves per call (registry load via get_system_prompt_text), so
  A/B-tested / GEPA-optimized registry versions are served immediately.
  """
  return Agent(
      name='BankingSupportAgent',
      instructions=get_system_prompt_text(),
      model=model or MODEL,
      tools=TOOLS,
  )


def _format_history(
    messages: list[dict[str, str]], current_user_message: str, customer_id: str
) -> list[dict[str, str]]:
  """Build the conversation list passed to the Runner.

  The customer-ID hint mirrors the app UI's customer selector: notebook
  questions phrased as "my account" resolve to the demo customer unless the
  caller supplies one.
  """
  hint = (
      f'The user is asking about customer ID **{customer_id}**. '
      'Use this customer ID when calling tools that require one unless the user '
      'specifies a different one.'
  )
  conv: list[dict[str, str]] = [{'role': 'system', 'content': hint}]
  conv.extend(messages)
  conv.append({'role': 'user', 'content': current_user_message})
  return conv


def _request_user_text(request: ResponsesAgentRequest) -> str:
  for item in request.input:
    item_dict = item.model_dump() if hasattr(item, 'model_dump') else item
    if item_dict.get('role') == 'user':
      content = item_dict.get('content')
      if isinstance(content, list) and len(content) > 0:
        first = content[0]
        return first.get('text') if isinstance(first, dict) else str(first)
      return str(content)
  return ''


def _build_output_item(text: str) -> 'TextOutputItem':
  """Shape the final answer so notebook 1's duck-parsing works
  (`item.content` -> parts with `.text`). Plain OutputItem stores content
  parts as dicts (no `.text` attribute), so the subclass wraps them."""
  return TextOutputItem(
      **create_text_output_item(text, id='msg_1')
  )


# ---------------------------------------------------------------------------
# The agent. `predict` is the notebook/eval interface (sync, one trace per
# call); `predict_stream` is the app's streaming interface (SSE shaping lives
# in the app layer).
# ---------------------------------------------------------------------------
def _run_coro_sync(coro) -> Any:
  """Run a coroutine to completion from ANY context and return its result.

  Databricks notebooks run an asyncio loop in the main thread, so bare
  asyncio.run() fails there ("cannot be called from a running event loop").
  Runs the coroutine on a private thread with its own fresh event loop, with
  the caller's contextvars COPIED into the thread so MLflow's trace context
  (started by the @mlflow.trace-decorated caller) follows the spans created
  inside the run.
  """
  box: dict = {}
  ctx = contextvars.copy_context()

  def _target() -> None:
    try:
      box['value'] = ctx.run(asyncio.run, coro)
    except BaseException as e:  # propagate to the caller verbatim
      box['error'] = e

  thread = threading.Thread(target=_target)
  thread.start()
  thread.join()
  if 'error' in box:
    raise box['error']
  return box['value']


class BankingAgent:
  """The banking customer-support agent (OpenAI Agents SDK under the hood)."""

  def __init__(self, model: Optional[str] = None):
    self.model = model or MODEL

  @mlflow.trace(span_type=SpanType.AGENT)
  def predict(
      self,
      request: ResponsesAgentRequest,
      *,
      session_id: Optional[str] = None,
      user_id: Optional[str] = None,
      conversation_history: Optional[list[dict]] = None,
      customer_id: Optional[str] = None,
  ) -> ResponsesAgentResponse:
    """Answer one agent request synchronously, under one MLflow trace.

    Used by the workshop notebooks and mlflow.genai.evaluate / GEPA. The
    Agents SDK Runner is async; the coroutine is drained via _run_coro_sync
    (a private thread + fresh event loop — Databricks notebooks run a loop
    in the main thread, so bare asyncio.run() fails there), with a FRESH
    OpenAI client bound to that short-lived loop (a shared client would keep
    connections on a previous, now-closed loop).
    """
    metadata = {}
    if session_id:
      metadata['mlflow.trace.session'] = session_id
    if user_id:
      metadata['mlflow.trace.user'] = user_id
    metadata['customer_id'] = customer_id or DEMO_CUSTOMER_ID
    if metadata:
      mlflow.update_current_trace(metadata=metadata)

    message = _request_user_text(request)
    mlflow.update_current_trace(request_preview=message)

    async def _drain() -> str:
      _configure_client()  # fresh client bound to THIS run's event loop
      agent = build_agent(self.model)
      conv = _format_history(
          conversation_history or [], message, customer_id or DEMO_CUSTOMER_ID
      )
      result = await Runner.run(agent, conv)
      return result.final_output

    final_text = _run_coro_sync(_drain())

    try:
      mlflow.update_current_trace(response_preview=final_text)
    except Exception:
      pass

    return ResponsesAgentResponse(
        output=[_build_output_item(final_text)],
        custom_outputs=request.custom_inputs,
    )

  async def predict_stream(
      self,
      message: str,
      customer_id: str,
      conversation_history: list[dict[str, str]],
  ) -> AsyncGenerator[Any, None]:
    """Stream Agents-SDK events for one chat turn (the app's SSE layer
    translates them into frontend events)."""
    agent = build_agent(self.model)
    conv = _format_history(conversation_history, message, customer_id)
    result = Runner.run_streamed(agent, input=conv)
    async for event in result.stream_events():
      yield event


AGENT = BankingAgent()


def get_agent() -> BankingAgent:
  """Get the singleton AGENT instance."""
  return AGENT
