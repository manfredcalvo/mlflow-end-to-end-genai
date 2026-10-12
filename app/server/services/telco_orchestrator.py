"""App streaming layer for the shared banking agent.

The agent itself (tools, prompt resolution, model routing, tracing setup)
lives in `mlflow_demo/agent/agent.py` — ONE implementation shared with the
workshop notebooks. This module keeps only the app-specific parts: per-request
MLflow experiment routing and the SSE event shaping for the frontend.

The agent runs DIRECTLY on the FastAPI process's event loop (async all the
way down — no asyncio.run/worker-thread loops). The original pattern drained
the Agents SDK Runner through asyncio.run per request, which closed each
loop with the shared client's connections/auth bound to it — the second
request per process then failed ("Event loop is closed") and auth degraded
flakily (403s). One persistent loop + one module-level client avoids the
entire class of bugs; @mlflow.trace supports async functions so the trace
still wraps the whole run.
"""

import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import mlflow
from mlflow.entities import SpanType

from mlflow_demo.agent.agent import AGENT

logger = logging.getLogger(__name__)


async def stream_telco_response(
  message: str,
  customer_id: str,
  conversation_history: list[dict[str, str]],
  experiment_id: str | None = None,
  experiment_path: str | None = None,
  model: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
  """Run the banking agent and yield SSE-shaped event dicts.

  Events match the frontend schema:
    - routing(agent_type, routing_decision)
    - tool_call(tool_name, call_id, arguments)
    - tool_result(call_id, output)
    - response_text(text)
    - completion(final_response, trace_id, agent_type, tools_used, done=True)
    - error(error, done=True)
  """
  yield {
    'type': 'routing',
    'agent_type': 'telco_supervisor',
    'routing_decision': f'Routed to BankingSupportAgent ({model or "shared default"})',
  }

  full_text_parts: list[str] = []
  tools_used: list[dict[str, Any]] = []

  try:
    if experiment_id:
      mlflow.set_experiment(experiment_id=experiment_id)
    elif experiment_path:
      mlflow.set_experiment(experiment_path)
    events, trace_id = await _traced_collect(message, customer_id, conversation_history)
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


@mlflow.trace(name='telco_chat', span_type=SpanType.AGENT)
async def _traced_collect(
  message: str,
  customer_id: str,
  conversation_history: list[dict[str, str]],
) -> tuple[list[Any], str | None]:
  """Drain the shared agent's stream under one MLflow trace span."""
  mlflow.update_current_trace(
    metadata={'mlflow.trace.session': customer_id, 'customer_id': customer_id},
    request_preview=message,
  )

  collected: list[Any] = []
  async for event in AGENT.predict_stream(message, customer_id, conversation_history):
    collected.append(event)

  trace_id = None
  try:
    trace_id = mlflow.get_last_active_trace_id()
  except Exception:
    pass

  # Surface the assistant's final text as the trace's response preview so
  # the experiment list shows useful summaries.
  final_text = ''
  for event in collected:
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
  return collected, trace_id


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
