"""Telco support agent routes.

Mounted under /api/telco/* to avoid collision with NFL DC Assistant routes.
Calls mlflow.set_experiment per-request inside the feedback handler so the
telco experiment stays isolated from the NFL one.
"""

import json
import logging
import traceback
from typing import Optional

import mlflow
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from mlflow.client import MlflowClient
from mlflow.entities import AssessmentSource, AssessmentSourceType
from pydantic import BaseModel, Field

from ..config.telco_settings import TelcoSettings, get_telco_settings
from ..services.telco_orchestrator import stream_telco_response  # noqa: F401

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/api/telco', tags=['telco'])


class ChatMessageModel(BaseModel):
  """Chat message model."""

  role: str = Field(..., description='Message role (user/assistant)')
  content: str = Field(..., description='Message content')


class ChatRequest(BaseModel):
  """Chat request model."""

  message: str = Field(..., description='User message')
  customer_id: str = Field(..., description='Customer ID, e.g. CUS-10001')
  conversation_history: list[ChatMessageModel] = Field(
    default_factory=list, description='Previous conversation messages'
  )


class AgentResponseModel(BaseModel):
  """Agent response model."""

  response: str
  agent_type: Optional[str] = None
  custom_outputs: Optional[dict] = None
  tools_used: Optional[list[dict]] = None
  trace_id: Optional[str] = None


class CustomerInfo(BaseModel):
  """Customer ID + display label."""

  customer_id: str
  display_name: str


class FeedbackRequest(BaseModel):
  """Feedback submission body."""

  trace_id: str = Field(..., description='MLflow trace ID')
  is_positive: bool = Field(..., description='Thumbs up = True, thumbs down = False')
  comment: Optional[str] = Field(None, description='Optional feedback comment')
  agent_id: Optional[str] = Field(
      None, description='Fallback reviewer ID; the logged-in user (X-Forwarded-Email) wins'
  )


class FeedbackResponse(BaseModel):
  """Feedback submission response."""

  status: str
  trace_id: str
  experiment_url: Optional[str] = None


@router.get('/customers', response_model=list[CustomerInfo])
async def get_demo_customers(settings: TelcoSettings = Depends(get_telco_settings)):
  """Return the demo customer list from env."""
  try:
    customers = []
    for customer_id in settings.demo_customer_ids:
      number = customer_id.replace('CUST-', '').replace('CUS-', '')
      customers.append(
        CustomerInfo(customer_id=customer_id, display_name=f'Customer {number}')
      )
    return customers
  except Exception as e:
    logger.error('Error getting demo customers: %s', e)
    logger.error(traceback.format_exc())
    raise HTTPException(status_code=500, detail=f'Error retrieving customers: {e}') from e


def _history_dicts(messages: list[ChatMessageModel]) -> list[dict]:
  return [{'role': m.role, 'content': m.content} for m in messages]


@router.post('/chat', response_model=AgentResponseModel)
async def chat(
  request: ChatRequest, settings: TelcoSettings = Depends(get_telco_settings)
):
  """Non-streaming chat completion that drains the in-app orchestrator stream."""
  try:
    final_response = ''
    tools_used: list[dict] = []
    trace_id: str | None = None
    agent_type: str | None = None
    async for evt in stream_telco_response(
      message=request.message,
      customer_id=request.customer_id,
      conversation_history=_history_dicts(request.conversation_history),
      experiment_id=settings.mlflow_experiment_id or None,
      experiment_path=settings.mlflow_experiment_path,
    ):
      if evt.get('type') == 'response_text':
        final_response = evt.get('text', '')
      elif evt.get('type') == 'tool_call':
        tools_used.append(evt)
      elif evt.get('type') == 'completion':
        final_response = evt.get('final_response', final_response)
        trace_id = evt.get('trace_id')
        agent_type = evt.get('agent_type')
        tools_used = evt.get('tools_used', tools_used)
      elif evt.get('type') == 'error':
        raise HTTPException(status_code=500, detail=evt.get('error', 'agent error'))
    return AgentResponseModel(
      response=final_response,
      agent_type=agent_type,
      tools_used=tools_used,
      trace_id=trace_id,
    )
  except HTTPException:
    raise
  except Exception as e:
    logger.error('Telco chat error: %s', e)
    logger.error(traceback.format_exc())
    raise HTTPException(status_code=500, detail=f'Chat processing error: {e}') from e


@router.post('/chat/stream')
async def chat_stream(
  request: ChatRequest, settings: TelcoSettings = Depends(get_telco_settings)
):
  """Streaming chat completion via Server-Sent Events."""
  try:
    async def event_generator():
      try:
        async for evt in stream_telco_response(
          message=request.message,
          customer_id=request.customer_id,
          conversation_history=_history_dicts(request.conversation_history),
          experiment_id=settings.mlflow_experiment_id or None,
          experiment_path=settings.mlflow_experiment_path,
        ):
          yield f'data: {json.dumps(evt)}\n\n'
      except Exception as e:
        logger.error('Streaming generator error: %s', e)
        logger.error(traceback.format_exc())
        yield f'data: {json.dumps({"type": "error", "error": str(e), "done": True})}\n\n'

    return StreamingResponse(
      event_generator(),
      media_type='text/event-stream',
      headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control',
      },
    )
  except Exception as e:
    logger.error('Streaming setup error: %s', e)
    logger.error(traceback.format_exc())
    raise HTTPException(status_code=500, detail=f'Error setting up streaming request: {e}') from e


@router.get('/health')
async def agent_health():
  """Lightweight liveness check for the in-app telco orchestrator."""
  return {'status': 'healthy', 'mode': 'in-app-agent'}


@router.get('/mlflow-experiment')
async def get_mlflow_experiment_info(settings: TelcoSettings = Depends(get_telco_settings)):
  """Return the telco MLflow experiment URL for the trace-link UI."""
  try:
    experiment_id = settings.mlflow_experiment_id
    mlflow_host = settings.databricks_host.rstrip('/')
    return {
      'experiment_id': experiment_id,
      'experiment_path': settings.mlflow_experiment_path,
      'mlflow_url': f'{mlflow_host}/ml/experiments/{experiment_id}' if experiment_id else None,
      'environment': settings.environment,
    }
  except Exception as e:
    logger.error('MLflow experiment info error: %s', e)
    logger.error(traceback.format_exc())
    return {'error': str(e), 'experiment_path': settings.mlflow_experiment_path, 'mlflow_url': None}


@router.post('/feedback', response_model=FeedbackResponse)
async def submit_feedback(
  request: FeedbackRequest,
  settings: TelcoSettings = Depends(get_telco_settings),
  # Injected by the Databricks Apps proxy: the email of the logged-in user
  # actually using the app. Falls back to the body's agent_id (legacy) or
  # 'unknown' when the header is absent (e.g. local dev).
  x_forwarded_email: Optional[str] = Header(None, alias='X-Forwarded-Email'),
):
  """Log thumbs-up/down feedback against a telco trace.

  Sets the telco MLflow experiment per request so traces don't cross-pollinate
  with the NFL experiment.
  """
  try:
    mlflow.set_tracking_uri('databricks')
    mlflow.set_experiment(settings.mlflow_experiment_path)

    client = MlflowClient()
    experiment_url = None
    try:
      trace = client.get_trace(request.trace_id)
      experiment_id = trace.info.experiment_id
      host = settings.databricks_host.rstrip('/')
      experiment_url = f'{host}/ml/experiments/{experiment_id}'
    except Exception as e:
      logger.warning('Could not resolve experiment URL from trace: %s', e)

    reviewer = x_forwarded_email or request.agent_id or 'unknown'
    mlflow.log_feedback(
      trace_id=request.trace_id,
      name='user_feedback',
      value=request.is_positive,
      source=AssessmentSource(
        source_type=AssessmentSourceType.HUMAN,
        source_id=reviewer,
      ),
      rationale=request.comment,
    )

    return FeedbackResponse(
      status='success',
      trace_id=request.trace_id,
      experiment_url=experiment_url,
    )
  except Exception as e:
    logger.error('Feedback submission error: %s', e)
    logger.error(traceback.format_exc())
    raise HTTPException(status_code=500, detail=f'Error submitting feedback: {e}') from e


