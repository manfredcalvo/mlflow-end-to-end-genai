"""In-process telco support orchestrator using the OpenAI Agents SDK.

Replaces the previous httpx -> Databricks model-serving endpoint round-trip
with a Databricks-hosted LLM (default: claude-sonnet-4-5) calling the 8 UC
functions copied to main.austin_choi_demo as tools. This follows the
Databricks multi-agent-apps pattern: one Agent runs in the FastAPI process,
MLflow autolog traces every step.

Tracing pattern matches the NFL agent (mlflow_demo/agent/agent.py): autolog
is registered at module import time so the patched OpenAI client is the one
the Agents SDK Runner uses, and the @mlflow.trace decorator wraps a sync
entry point so the whole agent run nests under one trace. The async event
streaming is run inside that sync wrapper via asyncio.run.
"""

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import mlflow
from agents import (
  Agent,
  Runner,
  set_default_openai_api,
  set_default_openai_client,
)
from agents.tracing import set_trace_processors
from databricks_openai import AsyncDatabricksOpenAI
from mlflow.entities import SpanType

from .telco_tools import TELCO_TOOLS

logger = logging.getLogger(__name__)

# Initialize at module import time so the OpenAI client used by Agents SDK is
# the one mlflow.openai.autolog has patched. Doing this lazily caused traces
# to skip telco runs entirely.
set_default_openai_client(AsyncDatabricksOpenAI())
set_default_openai_api('chat_completions')
set_trace_processors([])  # let MLflow be the only trace processor
mlflow.openai.autolog()
logging.getLogger('mlflow.utils.autologging_utils').setLevel(logging.ERROR)


SYSTEM_INSTRUCTIONS = """You are the customer support agent for a retail bank.
You help customer-service reps quickly answer questions about a specific customer
on the call. Be concise, factual, and friendly.

You have tools that read from the bank's databases (customer profile, accounts,
transactions, loans, credit cards, products, branches). Always use the tools when
the user asks about a specific customer's data — do not make up account details.

When the rep gives you a customer ID like CUST-00001:
- Use get_customer_profile first to load the customer's profile and tier.
- For account balances, use get_account_balance.
- For transaction history, use get_transaction_history (optionally a day range).
- For loan questions, use get_loan_details.
- For credit card questions, use get_credit_card_info.
- For product recommendations, use get_product_catalog (filter by category).
- For branch locations, use get_branch_info (filter by city).

Format dollar amounts with $ and dates as YYYY-MM-DD. If a tool returns no
rows or an error, tell the user briefly and offer next steps. Never expose
sensitive information the user did not ask for, and never guess data the tools
didn't return."""


# The live app serves the UC-registry prompt (<catalog>.<schema>.<prefix>_bank_support_prompt)
# when it exists — that's the prompt notebook 6's GEPA optimizes, so the Phase-4
# `bundle run bank_agent` redeploy picks up the improved version. The app's service
# principal cannot CREATE registry objects, so until a user context registers the
# prompt (the first workshop notebook does), we fall back to the built-in text above.
from mlflow_demo.utils.mlflow_helpers import load_or_register_prompt

SYSTEM_PROMPT_TEXT, SYSTEM_PROMPT_SOURCE = load_or_register_prompt(
    os.getenv('UC_CATALOG', ''),
    os.getenv('UC_SCHEMA', ''),
    os.getenv('PROMPT_NAME', ''),
    SYSTEM_INSTRUCTIONS,
    register_if_missing=False,
)
logger.info('System prompt source: %s', SYSTEM_PROMPT_SOURCE)


def _build_agent(model: str) -> Agent:
  """Build the telco support Agent with all 8 UC function tools."""
  return Agent(
    name='TelcoSupportAgent',
    instructions=SYSTEM_PROMPT_TEXT,
    model=model,
    tools=TELCO_TOOLS,
  )


def _format_history(messages: list[dict[str, str]], current_user_message: str, customer_id: str) -> list[dict[str, str]]:
  """Build the conversation list passed to Runner.run_streamed."""
  hint = (
    f'The rep is currently working a call for customer ID **{customer_id}**. '
    'Use this customer ID when calling tools that require one unless the rep '
    'specifies a different one.'
  )
  conv: list[dict[str, str]] = [{'role': 'system', 'content': hint}]
  conv.extend(messages)
  conv.append({'role': 'user', 'content': current_user_message})
  return conv


async def stream_telco_response(
  message: str,
  customer_id: str,
  conversation_history: list[dict[str, str]],
  experiment_id: str | None = None,
  experiment_path: str | None = None,
  model: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
  """Run the telco agent and yield SSE-shaped event dicts.

  Events match the legacy frontend schema so the existing telco UI
  (TelcoChat, TelcoIntelligentPanel) continues to work:
    - routing(agent_type, routing_decision)
    - tool_call(tool_name, call_id, arguments)
    - tool_result(call_id, output)
    - response_text(text)
    - completion(final_response, trace_id, agent_type, tools_used, done=True)
    - error(error, done=True)

  The Agents SDK Runner spawns its own asyncio task for the streaming loop;
  that task runs outside whatever async-generator context we set up here, so
  any mlflow context manager opened in this generator wouldn't reach the
  Runner's task. Workaround: run the entire agent in a worker thread inside
  a @mlflow.trace decorated sync function, then yield the collected events
  to the SSE consumer.
  """
  selected_model = model or os.getenv('TELCO_MODEL', 'databricks-claude-sonnet-4-5')

  yield {
    'type': 'routing',
    'agent_type': 'telco_supervisor',
    'routing_decision': f'Routed to TelcoSupportAgent ({selected_model})',
  }

  full_text_parts: list[str] = []
  tools_used: list[dict[str, Any]] = []

  try:
    events, trace_id = await asyncio.to_thread(
      _run_telco_traced,
      message,
      customer_id,
      conversation_history,
      selected_model,
      experiment_id,
      experiment_path,
    )
  except Exception as e:
    logger.exception('telco orchestrator run failed')
    yield {'type': 'error', 'error': str(e), 'done': True}
    return

  for event in events:
    async for shaped in _shape_event(event, full_text_parts, tools_used):
      yield shaped

  yield {
    'type': 'completion',
    'agent_type': 'telco_supervisor',
    'routing_decision': None,
    'tools_used': tools_used,
    'final_response': ''.join(full_text_parts),
    'trace_id': trace_id,
    'done': True,
  }


def _run_telco_traced(
  message: str,
  customer_id: str,
  conversation_history: list[dict[str, str]],
  selected_model: str,
  experiment_id: str | None,
  experiment_path: str | None,
) -> tuple[list[Any], str | None]:
  """Sync entry point that owns the MLflow trace context for one chat turn.

  Runs in a worker thread so the FastAPI event loop stays responsive. Sets
  the experiment, opens a fresh asyncio loop, drives the Agents SDK Runner
  to completion, collects every event, and returns (events, trace_id).
  """
  try:
    if experiment_id:
      mlflow.set_experiment(experiment_id=experiment_id)
    elif experiment_path:
      mlflow.set_experiment(experiment_path)
  except Exception as e:
    logger.warning('telco orchestrator: set_experiment failed: %s', e)

  events = _traced_collect(message, customer_id, conversation_history, selected_model)
  trace_id = None
  try:
    trace_id = mlflow.get_last_active_trace_id()
  except Exception:
    pass
  return events, trace_id


@mlflow.trace(name='telco_chat', span_type=SpanType.AGENT)
def _traced_collect(
  message: str,
  customer_id: str,
  conversation_history: list[dict[str, str]],
  selected_model: str,
) -> list[Any]:
  """Drive the Agents SDK to completion under one MLflow trace span."""
  mlflow.update_current_trace(
    metadata={'mlflow.trace.session': customer_id, 'customer_id': customer_id},
    request_preview=message,
  )

  async def _drain() -> list[Any]:
    agent = _build_agent(selected_model)
    conv = _format_history(conversation_history, message, customer_id)
    result = Runner.run_streamed(agent, input=conv)
    collected: list[Any] = []
    async for event in result.stream_events():
      collected.append(event)
    return collected

  events = asyncio.run(_drain())
  # Surface the assistant's final text as the trace's response preview so
  # the experiment list shows useful summaries.
  final_text = ''
  for event in events:
    if getattr(event, 'type', None) == 'raw_response_event':
      data = getattr(event, 'data', None)
      raw = data.model_dump() if data is not None and hasattr(data, 'model_dump') else None
      if raw and raw.get('type') == 'response.output_text.delta':
        final_text += raw.get('delta') or ''
  if final_text:
    try:
      mlflow.update_current_trace(response_preview=final_text)
    except Exception:
      pass
  return events


async def _shape_event(event: Any, text_parts: list[str], tools_used: list[dict[str, Any]]):
  """Translate one Agents-SDK StreamEvent into 0+ frontend SSE events."""
  etype = getattr(event, 'type', None)

  # Token-level deltas come through as raw_response_event with response.output_text.delta
  if etype == 'raw_response_event':
    data = getattr(event, 'data', None)
    if data is None:
      return
    raw = data.model_dump() if hasattr(data, 'model_dump') else dict(data)
    raw_type = raw.get('type')

    if raw_type == 'response.output_text.delta':
      delta = raw.get('delta') or ''
      if delta:
        text_parts.append(delta)
        yield {'type': 'response_text', 'text': ''.join(text_parts)}

    elif raw_type == 'response.output_item.added':
      item = raw.get('item', {}) or {}
      if item.get('type') == 'function_call':
        call_id = item.get('call_id') or item.get('id') or str(uuid.uuid4())
        info = {
          'type': 'tool_call',
          'tool_name': item.get('name', 'unknown'),
          'call_id': call_id,
          'arguments': item.get('arguments', '{}'),
        }
        tools_used.append(info)
        yield info

  # Tool outputs come as run_item_stream_event with tool_call_output_item
  elif etype == 'run_item_stream_event':
    item = getattr(event, 'item', None)
    if item is None:
      return
    item_type = getattr(item, 'type', None)
    if item_type == 'tool_call_output_item':
      raw_output = getattr(item, 'output', '')
      output_str = raw_output if isinstance(raw_output, str) else json.dumps(raw_output)
      call_id = getattr(item, 'call_id', None) or getattr(getattr(item, 'raw_item', None), 'call_id', None) or ''
      # Attach output back to the matching tool_call entry so the completion
      # event has the full pair.
      for tc in tools_used:
        if tc.get('call_id') == call_id:
          tc['output'] = output_str
          break
      yield {'type': 'tool_result', 'call_id': call_id, 'output': output_str}
