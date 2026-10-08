import json
import os
from pathlib import Path
from typing import Any, Callable, Generator, Optional
from uuid import uuid4
import warnings

import mlflow
import openai
from databricks.sdk import WorkspaceClient
from databricks_openai import UCFunctionToolkit, VectorSearchRetrieverTool
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    Message,
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from openai import OpenAI
from pydantic import BaseModel
from unitycatalog.ai.core.base import get_uc_function_client

def get_dbutils():
    try:
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        dbutils = DBUtils(spark)
        return dbutils
    except (ImportError, Exception):
        # Spark not available (e.g., MLflow validation environment)
        try:
            import IPython
            ipython = IPython.get_ipython()
            if ipython and "dbutils" in ipython.user_ns:
                return ipython.user_ns["dbutils"]
        except (ImportError, AttributeError, RuntimeError):
            pass
    return None  # dbutils not available
    
dbutils = get_dbutils()

# ============================================================================
# Load shared configuration from multiple sources (in priority order):
# 1. config/dc_assistant.json file (base config, including tools)
# 2. DC_ASSISTANT_CONFIG_JSON environment variable
# 3. Individual environment variables (set by the DAB app config, or by
#    setup_databricks_notebook_env() in the workshop notebooks)
#
# Environment variables (UC_CATALOG, UC_SCHEMA, PROMPT_NAME, LLM_MODEL,
# UC_TOOL_SCHEMA, UC_TOOL_PREFIX) always override the base config when set.
# The JSON file holds tool definitions and neutral defaults; the environment
# supplies the per-participant workspace values.
# ============================================================================
def load_config() -> dict:
    """Load configuration from file, JSON env var, or individual env vars.

    Environment variables UC_CATALOG, UC_SCHEMA, PROMPT_NAME, and LLM_MODEL
    override corresponding values in the base config when present.
    """
    config = None

    # Priority 1: Config file (base config with tools, auth defaults, etc.)
    config_path = Path(__file__).parent / "config" / "dc_assistant.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())

    # Priority 2: JSON config env var
    if config is None:
        config_env = os.getenv("DC_ASSISTANT_CONFIG_JSON")
        if config_env:
            config = json.loads(config_env)

    # Priority 3: Build minimal config from env vars if no file/JSON found
    if config is None:
        prompt_name = os.getenv("PROMPT_NAME")
        llm_model = os.getenv("LLM_MODEL")
        if not (prompt_name and llm_model):
            raise FileNotFoundError(
                "Configuration not found. Set one of: "
                "config/dc_assistant.json file, DC_ASSISTANT_CONFIG_JSON env var, "
                "or PROMPT_NAME + LLM_MODEL env vars"
            )
        config = {
            "workspace": {"catalog": "", "schema": ""},
            "prompt_registry": {"prompt_name": ""},
            "llm": {"endpoint_name": ""},
            "tools": {"uc_tool_names": []},
            "prompt_registry_auth": {
                "use_oauth": True,
                "secret_scope_name": os.getenv("SECRET_SCOPE_NAME", "dc-assistant-secrets"),
                "oauth_client_id_key": os.getenv("OAUTH_CLIENT_ID_KEY", "oauth-client-id"),
                "oauth_client_secret_key": os.getenv("OAUTH_CLIENT_SECRET_KEY", "oauth-client-secret"),
                "databricks_host": os.getenv("DATABRICKS_HOST", ""),
            },
        }

    # Apply env var overrides — these always win when set
    if os.getenv("UC_CATALOG"):
        config["workspace"]["catalog"] = os.environ["UC_CATALOG"]
    if os.getenv("UC_SCHEMA"):
        config["workspace"]["schema"] = os.environ["UC_SCHEMA"]
    if os.getenv("PROMPT_NAME"):
        config["prompt_registry"]["prompt_name"] = os.environ["PROMPT_NAME"]
    if os.getenv("LLM_MODEL"):
        config["llm"]["endpoint_name"] = os.environ["LLM_MODEL"]
    if os.getenv("DATABRICKS_HOST"):
        config.setdefault("prompt_registry_auth", {})["databricks_host"] = os.environ["DATABRICKS_HOST"]

    # Tool names: if UC_TOOL_SCHEMA is set, derive tool names from it using the
    # short names from the config file. This fully parameterizes tool resolution —
    # the catalog.schema prefix comes from env vars, not from the config file.
    # Env wins over the config file (same precedence as UC_CATALOG/UC_SCHEMA/etc.)
    # — the committed dc_assistant.json is a neutral template.
    tool_schema = os.getenv("UC_TOOL_SCHEMA") or config.get("tools", {}).get("uc_tool_schema")
    # UC_TOOL_PREFIX namespaces the functions per participant (e.g. jane_doe_get_...),
    # matching the <prefix>_get_* functions created by the data job in the shared schema.
    tool_prefix = os.getenv("UC_TOOL_PREFIX", "")
    p = f"{tool_prefix}_" if tool_prefix else ""
    if tool_schema:
        short_names = [
            name.split(".")[-1] for name in config["tools"].get("uc_tool_names", [])
        ]
        config["tools"]["uc_tool_names"] = [
            f"{tool_schema}.{p}{short}" for short in short_names
        ]

    return config

CONFIG = load_config()

PROMPT_NAME = CONFIG["prompt_registry"]["prompt_name"]
LLM_ENDPOINT_NAME = CONFIG["llm"]["endpoint_name"]
# Optional AI Gateway "model service": fully-qualified <catalog>.<schema>.<id>.
# When set, the agent routes LLM calls through the Unity Gateway model service
# (OpenAI-compatible endpoint at /ai-gateway/mlflow/v1) by its FQN, instead of a
# plain serving endpoint. Env wins; the agent config may also provide it.
LLM_MODEL_SERVICE = os.getenv("LLM_MODEL_SERVICE") or CONFIG.get("llm", {}).get("model_service", "")
UC_TOOL_NAMES = CONFIG["tools"].get("uc_tool_names", [])
UC_CATALOG = CONFIG["workspace"]["catalog"]
UC_SCHEMA = CONFIG["workspace"]["schema"]
SECRET_SCOPE_NAME = CONFIG.get("prompt_registry_auth", {}).get("secret_scope_name", "dc-assistant-secrets")
CLIENT_ID_KEY = CONFIG.get("prompt_registry_auth", {}).get("oauth_client_id_key", "oauth-client-id")
CLIENT_SECRET_KEY = CONFIG.get("prompt_registry_auth", {}).get("oauth_client_secret_key", "oauth-client-secret")
DATABRICKS_HOST = CONFIG.get("prompt_registry_auth", {}).get("databricks_host", "")

############################################
# Configure authentication for Prompt Registry access
# Priority order:
# 1. Databricks Apps run_as service principal (automatic - no secrets needed)
# 2. Environment variables (DATABRICKS_CLIENT_ID/SECRET or DATABRICKS_TOKEN)
# 3. Workspace secrets (fallback for notebook runs)
# See: https://docs.databricks.com/aws/en/generative-ai/agent-framework/agent-authentication
############################################

def setup_authentication():
    """Set up Databricks authentication based on available credentials."""
    # Set host if configured but not in environment
    if DATABRICKS_HOST and not os.getenv("DATABRICKS_HOST"):
        host = DATABRICKS_HOST.rstrip("/")
        os.environ["DATABRICKS_HOST"] = host

    # Databricks Apps may provide both DATABRICKS_TOKEN and DATABRICKS_CLIENT_ID/SECRET.
    # The SDK raises "more than one authorization method configured" if both are set.
    has_token = bool(os.getenv("DATABRICKS_TOKEN"))
    has_oauth = bool(os.getenv("DATABRICKS_CLIENT_ID") and os.getenv("DATABRICKS_CLIENT_SECRET"))
    if has_token and has_oauth:
        os.environ.pop("DATABRICKS_CLIENT_ID", None)
        os.environ.pop("DATABRICKS_CLIENT_SECRET", None)

    # Check if we're running in Databricks notebook
    if os.getenv("DATABRICKS_RUNTIME_VERSION") or os.getenv("DB_IS_DRIVER"):
        return WorkspaceClient()

    # Use default credential chain (reads env vars automatically)
    return WorkspaceClient()

WORKSPACE_CLIENT = setup_authentication()

# Force MLflow tracking through DatabricksTracingRestStore so traces honor
# the experiment's databricksTraceDestinationPath UC tag. Without this, MLflow
# silently falls back to legacy DBFS trace storage.
mlflow.set_tracking_uri("databricks")
print(f"[agent.py] MLflow tracking URI: {mlflow.get_tracking_uri()}")

# Configure MLflow to use Unity Catalog registry
mlflow.set_registry_uri("databricks-uc")

############################################
# Define your LLM endpoint and system prompt
############################################
# Single "ensure prompt" path (mlflow_helpers.load_or_register_prompt):
# - the UC registry has <catalog>.<schema>.<prompt_name> -> use it (what
#   notebook 6's GEPA optimizes, and what the app serves after a restart);
# - missing -> register ORIGINAL_PROMPT_TEMPLATE as v1 with the `production`
#   alias (user contexts hold CREATE rights; the first agent import registers);
# - can't register (e.g. the app SP) -> fall back to the built-in template.
from mlflow_demo.agent.prompts import ORIGINAL_PROMPT_TEMPLATE
from mlflow_demo.utils.mlflow_helpers import load_or_register_prompt

SYSTEM_PROMPT_TEXT, SYSTEM_PROMPT_SOURCE = load_or_register_prompt(
    UC_CATALOG, UC_SCHEMA, PROMPT_NAME, ORIGINAL_PROMPT_TEMPLATE, register_if_missing=True
)

class SYSTEM_PROMPT:
    """Duck-types the loaded Prompt object: .format(**kwargs) -> prompt text."""
    @staticmethod
    def format(**kwargs):
        return SYSTEM_PROMPT_TEXT


###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/generative-ai/agent-framework/agent-tool.html
###############################################################################
class ToolInfo(BaseModel):
    """
    Class representing a tool for the agent.
    - "name" (str): The name of the tool.
    - "spec" (dict): JSON description of the tool (matches OpenAI Responses format)
    - "exec_fn" (Callable): Function that implements the tool logic
    """

    name: str
    spec: dict
    exec_fn: Callable


def create_tool_info(tool_spec, exec_fn_param: Optional[Callable] = None):
    tool_spec["function"].pop("strict", None)
    tool_name = tool_spec["function"]["name"]
    udf_name = tool_name.replace("__", ".")

    # Define a wrapper that accepts kwargs for the UC tool call,
    # then passes them to the UC tool execution client. Drop None-valued
    # entries so the LLM passing `null` for an optional param falls through
    # to the function's DEFAULT instead of erroring with a type mismatch.
    def exec_fn(**kwargs):
        cleaned = {k: v for k, v in kwargs.items() if v is not None}
        function_result = uc_function_client.execute_function(udf_name, cleaned)
        if function_result.error is not None:
            return function_result.error
        else:
            return function_result.value
    return ToolInfo(name=tool_name, spec=tool_spec, exec_fn=exec_fn_param or exec_fn)


TOOL_INFOS = []

uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
uc_function_client = get_uc_function_client()
for tool_spec in uc_toolkit.tools:
    TOOL_INFOS.append(create_tool_info(tool_spec))


# Use Databricks vector search indexes as tools
# See [docs](https://docs.databricks.com/generative-ai/agent-framework/unstructured-retrieval-tools.html) for details

# # (Optional) Use Databricks vector search indexes as tools
# # See https://docs.databricks.com/generative-ai/agent-framework/unstructured-retrieval-tools.html


def _merge_parallel_tool_calls(chat_msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse consecutive assistant tool-call messages into a single one.

    `mlflow.types.responses.to_chat_completions_input` emits one
    `assistant {tool_calls: [...]}` per `function_call` item. When the LLM
    emits parallel tool calls (multiple tool_use blocks in one response),
    that produces back-to-back assistant messages — which Claude rejects
    on the next round-trip with "tool_use ids were found without
    tool_result blocks immediately after". Merge them so the on-the-wire
    conversation stays valid.
    """
    merged: list[dict[str, Any]] = []
    for msg in chat_msgs:
        if (
            merged
            and msg.get("role") == "assistant"
            and merged[-1].get("role") == "assistant"
            and msg.get("tool_calls")
            and merged[-1].get("tool_calls")
        ):
            merged[-1] = dict(merged[-1])
            merged[-1]["tool_calls"] = [*merged[-1]["tool_calls"], *msg["tool_calls"]]
            continue
        merged.append(msg)
    return merged


def _safe_parse_tool_arguments(raw_args: Any) -> dict:
    """Parse tool call arguments robustly.
    - Accepts dict (returns as-is) or JSON string.
    - Tolerates concatenated JSON objects (e.g., '}{') by selecting the last object.
    """
    if isinstance(raw_args, dict):
        return raw_args
    if not isinstance(raw_args, str):
        return {}
    s = raw_args.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Try to handle concatenated JSON objects like: '{...}{...}'
        if "}{" in s:
            # Try last object first
            try:
                return json.loads("{" + s.split("}{")[-1])
            except Exception:
                pass
            # Try as a list of objects and take the last one
            try:
                arr = json.loads("[" + s.replace("}{", "},{") + "]")
                if isinstance(arr, list) and arr:
                    return arr[-1]
            except Exception:
                pass
        # As a fallback, attempt to extract the first balanced JSON object
        depth = 0
        end = -1
        for i, ch in enumerate(s):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > 0:
            try:
                return json.loads(s[:end])
            except Exception:
                pass
        # Give up and raise a clean error message
        raise json.JSONDecodeError("Unable to parse tool arguments", s, 0)


def _to_chat_messages_with_images(items: list[dict]) -> list[dict[str, Any]]:
    """Convert Responses-API input items to chat completions messages, preserving images.

    mlflow.types.responses.to_chat_completions_input() hardcodes `content["text"]`
    for every part of a list-content message, so it raises KeyError on an
    `input_image` part instead of passing it through. Messages with an image
    part are converted here directly into the chat-completions multi-part
    format (text + image_url); everything else still goes through the
    library's to_chat_completions_input() unchanged.
    """
    messages: list[dict[str, Any]] = []
    for item in items:
        content = item.get("content") if isinstance(item, dict) else None
        has_image = isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") in ("input_image", "image_url")
            for part in content
        )
        if not has_image:
            messages.extend(to_chat_completions_input([item]))
            continue

        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in ("input_text", "text"):
                parts.append({"type": "text", "text": part.get("text", "")})
            elif part_type in ("input_image", "image_url"):
                url = part.get("image_url")
                if isinstance(url, dict):
                    url = url.get("url")
                parts.append({"type": "image_url", "image_url": {"url": url}})
        messages.append({"role": item.get("role", "user"), "content": parts})
    return messages


class ToolCallingAgent(ResponsesAgent):
    """
    Tool-calling Agent. Stateless per call -- session id, user id, and
    conversation history are passed explicitly to predict()/predict_stream()
    rather than stored on self, so one shared instance is safe to reuse
    concurrently across sessions/users (see server/routes/dc_assistant.py's
    SessionStore for where the actual per-session state lives).
    """

    def __init__(
        self,
        llm_endpoint: str,
        tools: list[ToolInfo],
        workspace_client: Optional[WorkspaceClient] = None,
    ):
        """Initializes the ToolCallingAgent with tools."""
        self.llm_endpoint = llm_endpoint
        self.workspace_client = workspace_client or WorkspaceClient()
        base_client: OpenAI = self.workspace_client.serving_endpoints.get_open_ai_client()
        if LLM_MODEL_SERVICE:
            # Route through the AI Gateway model service. It lives at a different
            # base path than /serving-endpoints; reuse the SDK client's auth via
            # with_options (preserves token handling) and call it by its FQN.
            host = self.workspace_client.config.host.rstrip("/")
            self.model_serving_client = base_client.with_options(
                base_url=f"{host}/ai-gateway/mlflow/v1"
            )
            self.llm_endpoint = LLM_MODEL_SERVICE
        else:
            self.model_serving_client = base_client
        self._tools_dict = {tool.name: tool for tool in tools}

    def get_tool_specs(self) -> list[dict]:
        """Returns tool specifications in the format OpenAI expects."""
        return [tool_info.spec for tool_info in self._tools_dict.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Executes the specified tool with the given arguments."""
        return self._tools_dict[tool_name].exec_fn(**args)

    def call_llm(
        self,
        messages: list[dict[str, Any]],
        model_serving_client: Optional[OpenAI] = None,
    ) -> Generator[dict[str, Any], None, None]:
        client = model_serving_client or self.model_serving_client
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
            for chunk in client.chat.completions.create(
                model=self.llm_endpoint,
                messages=_merge_parallel_tool_calls(to_chat_completions_input(messages)),
                tools=self.get_tool_specs(),
                stream=True,
            ):
                chunk_dict = chunk.to_dict()
                if len(chunk_dict.get("choices", [])) > 0:
                    yield chunk_dict

    def handle_tool_call(
        self,
        tool_call: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> ResponsesAgentStreamEvent:
        """
        Execute tool calls, add them to the running message history, and return a ResponsesStreamEvent w/ tool output
        """
        raw_args = tool_call.get("arguments", {})
        args = _safe_parse_tool_arguments(raw_args)
        result = str(self.execute_tool(tool_name=tool_call["name"], args=args))

        tool_call_output = self.create_function_call_output_item(tool_call["call_id"], result)
        messages.append(tool_call_output)
        return ResponsesAgentStreamEvent(type="response.output_item.done", item=tool_call_output)

    def call_and_run_tools(
        self,
        messages: list[dict[str, Any]],
        max_iter: int = 20,
        model_serving_client: Optional[OpenAI] = None,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        for _ in range(max_iter):
            last_msg = messages[-1]
            if last_msg.get("role", None) == "assistant":
                return
            elif last_msg.get("type", None) == "function_call":
                # When the LLM emits parallel tool calls, the aggregator
                # appends multiple consecutive function_call items. Process
                # ALL of them before calling the LLM again so each tool_use
                # has its tool_result, otherwise Claude rejects the next
                # request with "tool_use ids were found without tool_result
                # blocks immediately after".
                idx = len(messages)
                while idx > 0 and messages[idx - 1].get("type") == "function_call":
                    idx -= 1
                pending_calls = list(messages[idx:])
                for tool_call in pending_calls:
                    yield self.handle_tool_call(tool_call, messages)
            else:
                yield from output_to_responses_items_stream(
                    chunks=self.call_llm(messages, model_serving_client=model_serving_client),
                    aggregator=messages,
                )

        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item("Max iterations reached. Stopping.", str(uuid4())),
        )

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(
        self,
        request: ResponsesAgentRequest,
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        conversation_history: Optional[list[dict]] = None,
        model_serving_client: Optional[OpenAI] = None,
    ) -> ResponsesAgentResponse:
        # Only stamp session metadata when the caller is tracking a real
        # multi-turn session. Single-turn calls (evals, one-off questions)
        # must NOT get a session id — a session per trace clutters the
        # MLflow Sessions view with one-trace "conversations".
        metadata = {}
        if session_id:
            metadata["mlflow.trace.session"] = session_id
        if user_id:
            metadata["mlflow.trace.user"] = user_id
        if metadata:
            mlflow.update_current_trace(metadata=metadata)

        # Extract user input for clean display
        user_content = None
        for item in request.input:
            item_dict = item.model_dump() if hasattr(item, 'model_dump') else item
            if item_dict.get("role") == "user":
                content = item_dict.get("content")
                if isinstance(content, list) and len(content) > 0:
                    user_content = content[0].get("text") if isinstance(content[0], dict) else str(content[0])
                else:
                    user_content = str(content)
                break

        with mlflow.start_span(name="conversation_turn", span_type=SpanType.AGENT) as span:
            span.set_inputs({"request": user_content})

            outputs = [
                event.item
                for event in self.predict_stream(
                    request,
                    conversation_history=conversation_history,
                    model_serving_client=model_serving_client,
                )
                if event.type == "response.output_item.done"
            ]

            # Extract final assistant message for clean display
            assistant_response = None
            final_message = None
            for item in reversed(outputs):
                item_dict = item.model_dump() if hasattr(item, 'model_dump') else (item if isinstance(item, dict) else {})
                if item_dict.get("type") == "message" and item_dict.get("role") == "assistant":
                    final_message = item
                    content = item_dict.get("content", [])
                    if isinstance(content, list) and len(content) > 0:
                        if isinstance(content[0], dict) and "text" in content[0]:
                            assistant_response = content[0]["text"]
                            break

            if assistant_response:
                span.set_outputs({"response": assistant_response})

        return ResponsesAgentResponse(
            output=[final_message] if final_message else outputs,
            custom_outputs=request.custom_inputs
        )

    def predict_stream(
        self,
        request: ResponsesAgentRequest,
        *,
        conversation_history: Optional[list[dict]] = None,
        model_serving_client: Optional[OpenAI] = None,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        # Extract current user input
        current_input = [i.model_dump() for i in request.input]

        # Merge conversation history (if the caller passed one -- e.g. a
        # SessionStore-owned list for multi-turn) with the current request.
        # Mutating `conversation_history` in place below (rather than
        # replacing self.conversation_history) means different sessions'
        # calls never touch the same list object, so this is safe under
        # concurrency without any locking in the agent itself.
        history = conversation_history if conversation_history is not None else []
        all_input_items = list(history)
        all_input_items.extend(current_input)

        # Convert to chat completion format (preserves image content parts;
        # see _to_chat_messages_with_images)
        messages = _to_chat_messages_with_images(all_input_items)
        if SYSTEM_PROMPT:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT.format()})

        # Collect assistant response while streaming
        assistant_response = None

        for event in self.call_and_run_tools(messages=messages, model_serving_client=model_serving_client):
            yield event

            # Extract text response from message items
            if event.type == "response.output_item.done" and hasattr(event, 'item'):
                item = event.item
                if isinstance(item, dict):
                    if item.get("type") == "message" and item.get("role") == "assistant":
                        content = item.get("content", [])
                        if isinstance(content, list) and len(content) > 0:
                            if isinstance(content[0], dict) and "text" in content[0]:
                                assistant_response = content[0]["text"]

        # Save current turn to conversation history for next time
        history.extend(current_input)
        if assistant_response:
            history.append({
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_response}]
            })

    @mlflow.trace(name="dc_assistant_analysis")
    def predict_stream_local(
        self,
        question: str,
        conversation_history: Optional[list[dict]] = None,
        *,
        image_content: Optional[list[dict]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        model_serving_client: Optional[OpenAI] = None,
    ) -> Generator[dict, None, None]:
        """Stream a response for the given question (local convenience method).

        This wraps the ResponsesAgent predict_stream for easier use with FastAPI routes.

        Args:
            question: The user's question
            conversation_history: Optional list of prior conversation messages
                                 Format: [{"role": "user/assistant", "content": "..."}]
            image_content: Optional extra content parts (e.g. an `input_image`
                     part) to attach to THIS turn's user message alongside the
                     question text -- for the multi-modal tracing demo. This
                     call is always single-turn when image_content is set: it
                     builds one user message from question + image_content
                     and passes no conversation_history to predict_stream, so
                     the turn is never double-counted as both history and
                     current input (see agent.py predict_stream's
                     `all_input_items = history + current_input` merge).
                     Mutually exclusive with conversation_history.
            session_id: MLflow trace session id. Omit for single-turn calls —
                        the trace then carries no session metadata at all.
            user_id: Real end-user identity for trace attribution (e.g. from
                     an OBO-forwarded email), if available.
            model_serving_client: Per-request OpenAI client (e.g. OBO-scoped),
                                   defaults to the agent's service-principal client.

        Yields dicts with keys:
            - type: 'token', 'tool_call', 'done', or 'error'
            - content: token text (for type='token')
            - tool: tool info (for type='tool_call')
            - trace_id: MLflow trace ID (for type='done')
            - error: error message (for type='error')
        """
        # Build input messages - use conversation history if provided
        if image_content:
            # Single-turn multimodal call: build the one user message here and
            # forward no conversation_history, so predict_stream never merges
            # this turn's content with itself.
            input_messages = [
                Message(
                    role="user",
                    content=[{"type": "input_text", "text": question}, *image_content],
                )
            ]
            conversation_history = None
        elif conversation_history:
            # Use the full conversation history
            input_messages = [
                Message(role=msg["role"], content=msg["content"]) for msg in conversation_history
            ]
        else:
            # Single turn - just the current question
            input_messages = [Message(role="user", content=question)]

        request = ResponsesAgentRequest(input=input_messages)

        try:
            # Capture the trace ID from the @mlflow.trace span immediately,
            # before autolog creates nested spans that could change the context
            active_span = mlflow.get_current_active_span()
            trace_id = active_span.trace_id if active_span else None

            # Set clean request preview + user metadata for MLflow UI.
            # Session metadata only when the caller tracks a multi-turn
            # session — single-turn traces must not appear in Sessions view.
            metadata = {}
            if session_id:
                metadata["mlflow.trace.session"] = session_id
            if user_id:
                metadata["mlflow.trace.user"] = user_id
            mlflow.update_current_trace(request_preview=question, metadata=metadata or None)

            full_response = ''

            for event in self.predict_stream(
                request,
                conversation_history=conversation_history,
                model_serving_client=model_serving_client,
            ):
                if event.type == "response.output_item.done":
                    item = event.item
                    item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)

                    if item_type == "function_call":
                        item_dict = item if isinstance(item, dict) else item.model_dump()
                        yield {
                            "type": "tool_call",
                            "tool": {
                                "name": item_dict.get("name"),
                                "arguments": item_dict.get("arguments"),
                            },
                        }
                    elif item_type == "function_call_output":
                        pass  # Tool output - already handled
                    elif item_type == "message":
                        item_dict = item if isinstance(item, dict) else item.model_dump()
                        content = item_dict.get("content", [])
                        if content and isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict) and c.get("type") == "output_text":
                                    text = c.get("text", "")
                                    full_response += text
                                    yield {"type": "token", "content": text}

                elif event.type == "response.content_part.delta":
                    delta = event.delta if hasattr(event, "delta") else None
                    if delta:
                        text = delta.get("text") if isinstance(delta, dict) else getattr(delta, "text", None)
                        if text:
                            full_response += text
                            yield {"type": "token", "content": text}

            # Set clean response preview for MLflow UI
            if full_response:
                mlflow.update_current_trace(response_preview=full_response)

            # Fall back to get_last_active_trace_id if span wasn't available at start
            if not trace_id:
                trace_id = mlflow.get_last_active_trace_id()

            yield {"type": "done", "trace_id": trace_id}

        except Exception as e:
            yield {"type": "error", "error": str(e)}


# Log the model using MLflow
mlflow.openai.autolog()
AGENT = ToolCallingAgent(
    llm_endpoint=LLM_ENDPOINT_NAME, tools=TOOL_INFOS, workspace_client=WORKSPACE_CLIENT
)
mlflow.models.set_model(AGENT)


def get_agent() -> ToolCallingAgent:
    """Get the singleton AGENT instance."""
    return AGENT