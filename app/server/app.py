"""FastAPI app for the Lakehouse Apps + Agents demo."""

import argparse
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from mlflow_demo.utils.mlflow_helpers import get_mlflow_experiment_id
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from .routes import telco

# Configure logging for Databricks Apps monitoring
# Logs written to stdout/stderr will be available in Databricks Apps UI and /logz endpoint
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  handlers=[
    logging.StreamHandler(),  # This ensures logs go to stdout for Databricks Apps monitoring
  ],
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
  """Manage application lifespan events."""
  # Startup
  logger.info('Starting application...')

  # Don't call mlflow.tracing.set_destination at app startup. The experiment
  # is already linked to a specific UC trace table via its
  # databricksTraceDestinationPath tag; setting a process-global schema-level
  # destination overrides that link and routes traces to a non-existent
  # mlflow_experiment_trace_otel_spans table. Per-request mlflow.set_experiment
  # is enough — MLflow resolves the right destination from the experiment.

  logger.info('Application startup complete')

  yield

  # Shutdown
  logger.info('Application shutting down...')
  logger.info('Application shutdown complete')


app = FastAPI(lifespan=lifespan)

# Enable CORS for frontend to access backend APIs
app.add_middleware(
  CORSMiddleware,
  allow_origins=['*'],  # Change this to specific origins in production
  allow_credentials=True,
  allow_methods=['*'],
  allow_headers=['*'],
)

# We're in dev mode when the server has the reload bit.
IS_DEV = os.getenv('IS_DEV', 'false').lower() == 'true'
# Parse arguments at startup
parser = argparse.ArgumentParser()
# bool
parser.add_argument('--reload', action='store_true')
args, _ = parser.parse_known_args()  # Ignore unknown args
IS_DEV = args.reload

PORT = int(os.getenv('UVICORN_PORT', 8000))
HOST = os.getenv('UVICORN_HOST', '0.0.0.0')

API_PREFIX = '/api'

# Include route modules
app.include_router(telco.router)


# Common/shared models
class ExperimentInfo(BaseModel):
  """Experiment info."""

  experiment_id: str
  link: str
  trace_url_template: str
  session_url_template: str
  evaluation_run_url_template: str
  failed_traces_url: str
  eval_dataset_url: str
  monitoring_url: str


def ensure_https_protocol(host: str | None) -> str:
  """Ensure the host URL has HTTPS protocol and strip trailing slashes."""
  if not host:
    return ''

  # Add protocol if missing
  if not (host.startswith('https://') or host.startswith('http://')):
    host = f'https://{host}'

  # Strip trailing slashes to avoid double slashes when concatenating paths
  return host.rstrip('/')


# Common/shared endpoints
@app.get(f'{API_PREFIX}/tracing_experiment')
async def experiment():
  """Get the MLFlow experiment info."""
  databricks_host = ensure_https_protocol(os.getenv('DATABRICKS_HOST'))

  experiment_id = get_mlflow_experiment_id()
  workspace_id = os.getenv('DATABRICKS_WORKSPACE_ID', '')
  sql_warehouse_id = os.getenv('SQL_WAREHOUSE_ID', '')

  # Build session URL with optional workspace and warehouse params
  session_params = [f'sessionId={{sessionId}}']
  if workspace_id:
    session_params.append(f'o={workspace_id}')
  if sql_warehouse_id:
    session_params.append(f'sqlWarehouseId={sql_warehouse_id}')
  session_query = '&'.join(session_params)

  # Deep-link template for a specific evaluation run in the MLflow UI
  eval_run_url = (
    f'{databricks_host}/ml/experiments/{experiment_id}/evaluation-runs'
    f'?selectedRunUuid={{runId}}'
  )
  if workspace_id:
    eval_run_url += f'&o={workspace_id}'

  return ExperimentInfo(
    experiment_id=experiment_id,
    link=f'{databricks_host}/ml/experiments/{experiment_id}?compareRunsMode=TRACES',
    trace_url_template=f'{databricks_host}/ml/experiments/{experiment_id}/traces?selectedEvaluationId=',
    session_url_template=f'{databricks_host}/ml/experiments/{experiment_id}/chat-sessions/{{sessionId}}?{session_query}',
    evaluation_run_url_template=eval_run_url,
    failed_traces_url=f'{databricks_host}/ml/experiments/{experiment_id}/traces?&filter=TAG%3A%3A%3D%3A%3Ayes%3A%3Aeval_example&filter=ASSESSMENT%3A%3A%3D%3A%3Ano%3A%3Aaccuracy',
    eval_dataset_url=f'{databricks_host}/ml/experiments/{experiment_id}/datasets',
    monitoring_url=f'{databricks_host}/ml/experiments/{experiment_id}/evaluation-monitoring',
  )


@app.get(f'{API_PREFIX}/dashboard')
async def dashboard_embed():
  """Embed info for the bundle-created AI/BI portfolio dashboard.

  The frontend's Analytics tab iframes the returned embed_url; it renders with
  the viewer's existing workspace session (app users are already
  Databricks-authenticated).
  """
  databricks_host = ensure_https_protocol(os.getenv('DATABRICKS_HOST'))
  dashboard_id = os.getenv('DASHBOARD_ID', '')

  return {
      'dashboard_id': dashboard_id,
      'embed_url': (
          f'{databricks_host}/embed/dashboardsv3/{dashboard_id}'
          if dashboard_id
          else None
      ),
  }


@app.get(f'{API_PREFIX}/health')
async def health_check():
  """Health check endpoint for monitoring app status."""
  import time

  try:
    # Test MLflow connection
    experiment_id = get_mlflow_experiment_id()

    # Test basic functionality
    health_status = {
      'status': 'healthy',
      'timestamp': int(time.time() * 1000),
      'mlflow_experiment_id': experiment_id,
      'environment': 'production' if not IS_DEV else 'development',
    }

    logger.info('Health check passed - all systems operational')
    return health_status

  except Exception as e:
    logger.error(f'Health check failed: {str(e)}')
    return {
      'status': 'unhealthy',
      'error': str(e),
      'timestamp': int(time.time() * 1000),
    }


# Static file serving and dev proxy
if IS_DEV:
  # Development: Proxy non-API requests to frontend dev server on port 3000
  import httpx

  @app.get('/{full_path:path}')
  async def dev_proxy(full_path: str):
    """Proxy non-API requests to Vite dev server in development."""
    # Don't interfere with API routes
    if full_path.startswith('api/'):
      raise HTTPException(status_code=404, detail='API endpoint not found')

    # Proxy to Vite dev server
    frontend_url = f'http://localhost:3000/{full_path}'
    async with httpx.AsyncClient() as client:
      try:
        response = await client.get(frontend_url, follow_redirects=True)
        return Response(content=response.content, status_code=response.status_code, media_type=response.headers.get('content-type'))
      except Exception as e:
        logger.error(f'Failed to proxy to frontend: {e}')
        return Response(
          content=f'Frontend dev server not running. Make sure Vite is running on port 3000.\n\nError: {str(e)}',
          status_code=502,
          media_type='text/plain'
        )

else:
  # Production: Serve the built React files
  build_path = Path('.') / 'app/client/build'
  if build_path.exists():
    # Mount static assets first (these should not fallback to index.html)
    app.mount('/assets', StaticFiles(directory=build_path / 'assets'), name='assets')

    # Add catch-all route for SPA routing (must come after API routes)
    @app.get('/{full_path:path}')
    async def spa_fallback(full_path: str):
      """Serve static files from the build root, falling back to index.html for SPA routes."""
      # Don't interfere with API routes
      if full_path.startswith('api/'):
        raise HTTPException(status_code=404, detail='API endpoint not found')

      # Public assets (images, favicon, manifest.json, etc.) are copied to the
      # build root by the frontend build — serve them directly instead of
      # falling through to index.html, or every root-level asset 404s as HTML.
      requested_file = (build_path / full_path).resolve()
      if (
        full_path
        and requested_file.is_file()
        and build_path.resolve() in requested_file.parents
      ):
        return FileResponse(requested_file)

      index_file = build_path / 'index.html'
      if index_file.exists():
        return FileResponse(index_file)
      else:
        raise HTTPException(status_code=404, detail='Application not built')

  else:
    raise RuntimeError(f'Build directory {build_path} not found. Run `bun run build` in client/')


if __name__ == '__main__':
  uvicorn.run(app, host=HOST, port=PORT)
