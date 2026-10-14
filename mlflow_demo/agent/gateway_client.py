"""OpenAI client for the banking agent: the AI-Gateway model service with
Claude reasoning-block normalization.

Route through the participant's AI-Gateway model service when configured —
via the OpenAI-NATIVE gateway path ({host}/ai-gateway/openai/v1,
use_ai_gateway_native_api=True), which returns proper OpenAI-shaped deltas
(the mlflow/v1 path passes Anthropic content blocks through, which the
Agents SDK's strict event models reject) — else the plain serving endpoint.
"""

import os
from typing import Any

from databricks_openai import AsyncDatabricksOpenAI
from databricks_openai.utils.clients import (
  AsyncDatabricksChat,
  AsyncDatabricksCompletions,
)
from openai.resources.chat.chat import AsyncChat


# ---------------------------------------------------------------------------
# Reasoning-block normalization. Claude reasoning models (Sonnet 5)
# stochastically emit Anthropic-style CONTENT BLOCKS (e.g. {"type":
# "reasoning", ...}) inside the OpenAI-compat streaming delta.content — as a
# LIST where OpenAI clients expect a string. The OpenAI Agents SDK's strict
# event models reject that ("ResponseTextDeltaEvent ... Input should be a
# valid string"). Wrap the client's chat.completions to flatten any
# block-list content: keep text blocks' text, drop reasoning/redacted blocks.
# ---------------------------------------------------------------------------


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
  if os.getenv('LLM_MODEL_SERVICE', ''):
    return _GatewayOpenAI(use_ai_gateway_native_api=True)
  return _GatewayOpenAI()
