"""FastAPI dependency for per-request user identity + on-behalf-of auth.

Databricks Apps configured with `user_authorization` scopes (see app.yaml)
forward the end user's OAuth token via HTTP headers on each request. This
module reads those headers and builds a per-request WorkspaceClient scoped
to the real user, falling back to the app's service-principal client when no
forwarded token is present (local dev, or before user_authorization is
configured in app.yaml).
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from databricks.sdk import WorkspaceClient
from fastapi import Header
from openai import OpenAI

logger = logging.getLogger(__name__)

# Header names Databricks Apps injects when `user_authorization` scopes are
# configured. Verify against current Databricks Apps docs if these ever stop
# showing up in production.
_FORWARDED_TOKEN_HEADER = "X-Forwarded-Access-Token"
_FORWARDED_EMAIL_HEADER = "X-Forwarded-Email"
_FORWARDED_USER_HEADER = "X-Forwarded-User"

_default_client: Optional[WorkspaceClient] = None


def _default_workspace_client() -> WorkspaceClient:
    """Lazily build (once) the shared service-principal WorkspaceClient."""
    global _default_client
    if _default_client is None:
        _default_client = WorkspaceClient()
    return _default_client


@dataclass
class RequestUserContext:
    """Per-request identity and Databricks clients scoped to that identity."""

    user_email: Optional[str] = None
    user_id: Optional[str] = None
    is_obo: bool = False
    _obo_client: Optional[WorkspaceClient] = field(default=None, repr=False)

    @property
    def workspace_client(self) -> WorkspaceClient:
        return self._obo_client or _default_workspace_client()

    @property
    def model_serving_client(self) -> OpenAI:
        return self.workspace_client.serving_endpoints.get_open_ai_client()


def get_user_context(
    x_forwarded_access_token: Optional[str] = Header(None, alias=_FORWARDED_TOKEN_HEADER),
    x_forwarded_email: Optional[str] = Header(None, alias=_FORWARDED_EMAIL_HEADER),
    x_forwarded_user: Optional[str] = Header(None, alias=_FORWARDED_USER_HEADER),
) -> RequestUserContext:
    """FastAPI dependency: resolve the caller's identity and a WorkspaceClient
    scoped to them, when Databricks Apps OBO is configured.

    Always returns a usable context -- never raises -- so a request from
    local dev (no forwarded headers) or a stale/expired forwarded token
    degrades to the service-principal identity instead of failing.
    """
    user_id = x_forwarded_email or x_forwarded_user

    if not x_forwarded_access_token:
        return RequestUserContext(user_email=x_forwarded_email, user_id=user_id, is_obo=False)

    try:
        # auth_type="pat" is required here: Databricks Apps also sets
        # DATABRICKS_CLIENT_ID/SECRET (the app's own service-principal auth)
        # in the environment, and WorkspaceClient raises "more than one
        # authorization method configured" if both an explicit token and
        # ambient oauth env vars are present without this override.
        obo_client = WorkspaceClient(
            host=os.environ.get("DATABRICKS_HOST"), token=x_forwarded_access_token, auth_type="pat"
        )
        # WorkspaceClient(token=...) doesn't validate eagerly -- without this
        # cheap identity check, an expired/invalid token would only surface
        # as an error from the first real (and much more expensive) call,
        # e.g. a model-serving request -- and callers that don't already
        # catch that (like /multi-turn) would 500 instead of degrading.
        obo_client.current_user.me()
    except Exception as e:
        logger.warning(f"Forwarded token invalid, falling back to service principal: {e}")
        return RequestUserContext(user_email=x_forwarded_email, user_id=user_id, is_obo=False)

    return RequestUserContext(
        user_email=x_forwarded_email, user_id=user_id, is_obo=True, _obo_client=obo_client
    )
