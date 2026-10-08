"""MLflow utility functions for tracing and UI link generation."""

import logging
import os
import re
from typing import Optional


def sanitize_prefix(user: str) -> str:
  """Turn a username/email into a safe UC identifier prefix (jane.doe@x.com -> jane_doe)."""
  return re.sub(r'[^a-z0-9]', '_', user.split('@')[0].lower())


def get_dab_experiment_name(user_email: str) -> str:
  """The UNMANGLED experiment name from resources/bgp_experiment.yml.

  NOTE: dev-mode (`--target dev`) deploys rename it to
  '/Users/<you>/[dev <prefix>] bgp_bank_workshop', so do NOT look it up by this
  exact string — use setup_databricks_notebook_env()'s search-based resolution,
  or pass the exact name from the DAB (${resources.experiments...}) where the
  DAB can inject it (app env, job parameters).
  """
  return f'/Users/{user_email}/bgp_bank_workshop'


# Shared-workshop defaults surfaced as notebook widgets. The instructor sets these
# once per workspace; participants can override them per-run in the widget bar.
WIDGET_DEFAULTS = {
  'catalog': 'workshop_bgp',
  'schema': 'shared_data',
  'warehouse': 'd9ce200bcd25d8a2',
}


def _find_user_bgp_experiment(mlflow, user_email):
  """Find the bundle-created experiment for THIS user, robust to DAB name mangling.

  Dev-mode deploys rename the experiment (e.g. '/Users/<you>/[dev <prefix>]
  bgp_bank_workshop'), so an exact-name lookup fails. Search instead and take the
  active experiment whose path starts with /Users/<you>/.
  """
  exps = mlflow.search_experiments(
      filter_string="name LIKE '%bgp_bank_workshop'", max_results=100
  )
  mine = [
      e for e in exps
      if e.name.startswith(f'/Users/{user_email}/') and e.lifecycle_stage == 'active'
  ]
  if not mine:
    return None
  if len(mine) > 1:
    # Most recently updated wins (e.g. leftover experiments from re-deploys).
    mine.sort(key=lambda e: e.last_update_time or 0, reverse=True)
  return mine[0]


def setup_databricks_notebook_env():
  """Configure a workshop notebook from the CURRENT USER + shared widgets.

  Every per-participant value (object prefix, prompt name, model service) derives
  from the running user's email. The widgets carry the shared catalog/schema/
  warehouse, plus an optional `experiment_name` (jobs pass the exact name from
  the DAB; interactive runs leave it blank and we find the bundle-created
  experiment). Requires the bundle to have been deployed.
  """
  import os

  import mlflow
  from databricks.sdk import WorkspaceClient
  from databricks.sdk.runtime import dbutils

  user = WorkspaceClient().current_user.me().user_name
  prefix = sanitize_prefix(user)

  dbutils.widgets.text('catalog', WIDGET_DEFAULTS['catalog'], 'Shared UC catalog')
  dbutils.widgets.text('schema', WIDGET_DEFAULTS['schema'], 'Shared UC schema')
  dbutils.widgets.text('warehouse', WIDGET_DEFAULTS['warehouse'], 'SQL warehouse (tracing)')
  dbutils.widgets.text(
      'experiment_name', '', 'MLflow experiment (blank = find the one your bundle deploy created)'
  )
  catalog = dbutils.widgets.get('catalog')
  schema = dbutils.widgets.get('schema')
  warehouse = dbutils.widgets.get('warehouse')
  explicit_experiment = dbutils.widgets.get('experiment_name').strip()

  mlflow.set_tracking_uri('databricks')

  if explicit_experiment:
    exp = mlflow.get_experiment_by_name(explicit_experiment)
    if exp is None:
      raise RuntimeError(f"MLflow experiment '{explicit_experiment}' not found.")
  else:
    exp = _find_user_bgp_experiment(mlflow, user)
    if exp is None:
      raise RuntimeError(
          f'No MLflow experiment matching /Users/{user}/…bgp_bank_workshop found — '
          'run `databricks bundle deploy` first.'
      )

  # Everything else derives from catalog/schema/prefix — identical to the env the
  # DAB sets for the app (resources/bgp_agent_app.yml), so notebooks and the app
  # resolve the same UC objects and experiment.
  os.environ.update({
    'MLFLOW_TRACKING_URI': 'databricks',
    'UC_CATALOG': catalog,
    'UC_SCHEMA': schema,
    'UC_TOOL_SCHEMA': f'{catalog}.{schema}',
    'UC_TOOL_PREFIX': prefix,
    'PROMPT_NAME': f'{prefix}_bgp_support_prompt',
    'PROMPT_ALIAS': 'production',
    'SCORER_PREFIX': prefix,
    'LLM_MODEL': 'databricks-claude-sonnet-5',  # informational; the agent uses LLM_MODEL_SERVICE
    'LLM_MODEL_SERVICE': f'{catalog}.{schema}.{prefix}_bgp_agent_llm',
    'MLFLOW_TRACING_SQL_WAREHOUSE_ID': warehouse,
    'MLFLOW_ENABLE_ASYNC_TRACE_LOGGING': 'false',
    'MLFLOW_TRACE_EXTRACT_ATTACHMENTS': 'false',
    'MLFLOW_EXPERIMENT_ID': exp.experiment_id,
    'MLFLOW_EXPERIMENT_NAME': exp.name,
  })
  mlflow.set_experiment(experiment_id=exp.experiment_id)
  return exp


def load_or_register_prompt(catalog, schema, prompt_name, fallback_template,
                             register_if_missing=True):
  """Use the UC-registered prompt when available; register it when missing.

  The single "ensure prompt" path for every agent context:
  - Registry has it -> use it (the canonical prompt; what notebook 6's GEPA
    optimizes and what the app serves after a restart).
  - Missing + register_if_missing (user contexts, which hold CREATE rights) ->
    register `fallback_template` as v1 with the `production` alias.
  - Missing + can't register (e.g. the app's service principal) -> return the
    fallback text so the agent still runs; the registry becomes canonical once
    any user context registers it.

  Returns (template_text, source) with source in {'registry', 'registered', 'fallback'}.
  """
  import mlflow

  full = f'{catalog}.{schema}.{prompt_name}'
  try:
    prompt = mlflow.genai.load_prompt(f'prompts:/{full}')
    return prompt.template, 'registry'
  except Exception:
    pass  # not registered yet

  if register_if_missing:
    try:
      prompt = mlflow.genai.register_prompt(
          name=full,
          template=fallback_template,
          commit_message='Initial BGP banking customer-support system prompt',
      )
      mlflow.genai.set_prompt_alias(name=full, alias='production', version=prompt.version)
      # Tag the experiment so the MLflow UI links prompts for it (best-effort).
      try:
        mlflow.set_experiment_tags({'mlflow.promptRegistryLocation': f'{catalog}.{schema}'})
      except Exception:
        pass
      return prompt.template, 'registered'
    except Exception as e:
      logging.getLogger(__name__).warning('Could not register prompt %s: %s', full, e)

  logging.getLogger(__name__).warning(
      'Prompt %s not in the registry — using the built-in template.', full
  )
  return fallback_template, 'fallback'


def link_experiment_to_uc_schema():
  """Link the MLflow experiment to a Unity Catalog schema for trace storage.

  This is a one-time setup that creates three UC tables for storing trace data:
    - {catalog}.{schema}.mlflow_experiment_trace_otel_logs
    - {catalog}.{schema}.mlflow_experiment_trace_otel_metrics
    - {catalog}.{schema}.mlflow_experiment_trace_otel_spans

  Idempotent: safe to call on every notebook run. If already linked, this is a no-op.
  Requires UC_CATALOG, UC_SCHEMA, MLFLOW_EXPERIMENT_ID, and MLFLOW_TRACING_SQL_WAREHOUSE_ID.
  """
  import os

  import mlflow
  from mlflow.entities import UCSchemaLocation
  from mlflow.tracing.enablement import set_experiment_trace_location

  uc_catalog = os.getenv('UC_CATALOG')
  uc_schema = os.getenv('UC_SCHEMA')
  experiment_id = os.getenv('MLFLOW_EXPERIMENT_ID')
  warehouse_id = os.getenv('MLFLOW_TRACING_SQL_WAREHOUSE_ID')

  if not all([uc_catalog, uc_schema, experiment_id, warehouse_id]):
    return

  try:
    os.environ['MLFLOW_TRACING_SQL_WAREHOUSE_ID'] = warehouse_id
    set_experiment_trace_location(
      location=UCSchemaLocation(
        catalog_name=uc_catalog,
        schema_name=uc_schema,
      ),
      experiment_id=experiment_id,
    )
  except Exception:
    # Already linked or insufficient permissions — safe to ignore
    pass


def get_mlflow_experiment_id() -> Optional[str]:
  """Gets the current mlflow experiment id.

  If MLFLOW_EXPERIMENT_ID is set, returns it directly.
  Otherwise, if MLFLOW_EXPERIMENT_NAME is set, resolves the experiment by name
  (creating it if needed). This lets the DAB bundle define the experiment as a
  resource and pass its name — the ID is resolved at app startup.
  """
  exp_id = os.environ.get('MLFLOW_EXPERIMENT_ID')
  if exp_id:
    return exp_id

  exp_name = os.environ.get('MLFLOW_EXPERIMENT_NAME')
  if exp_name:
    import mlflow
    mlflow.set_tracking_uri('databricks')
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
      exp_id = mlflow.create_experiment(exp_name)
      logging.info(f'Created MLflow experiment: {exp_name} (ID: {exp_id})')
    else:
      exp_id = exp.experiment_id
      logging.info(f'Resolved MLflow experiment: {exp_name} (ID: {exp_id})')
    os.environ['MLFLOW_EXPERIMENT_ID'] = str(exp_id)
    return str(exp_id)

  return None


def ensure_https_protocol(host: str | None) -> str:
  """Ensure the host URL has HTTPS protocol."""
  if not host:
    return ''

  if host.startswith('https://') or host.startswith('http://'):
    return host

  return f'https://{host}'


def generate_trace_links(trace_id: str = None, print_urls: bool = True) -> tuple[str, str]:
  """Generate MLflow UI links for viewing traces."""
  import mlflow

  if mlflow.utils.databricks_utils.is_in_databricks_notebook():
    databricks_host = ensure_https_protocol(mlflow.utils.databricks_utils.get_browser_hostname())
  else:
    databricks_host = ensure_https_protocol(os.getenv('DATABRICKS_HOST'))

  experiment_id = os.getenv('MLFLOW_EXPERIMENT_ID')

  if not databricks_host or not experiment_id:
    print('⚠️ Missing DATABRICKS_HOST or MLFLOW_EXPERIMENT_ID - cannot generate links')
    return

  # General experiment traces view
  traces_url = f'{databricks_host}/ml/experiments/{experiment_id}?compareRunsMode=TRACES'

  # Specific trace view if trace_id provided
  specific_trace_url = None
  if trace_id:
    specific_trace_url = (
      f'{databricks_host}/ml/experiments/{experiment_id}/traces?selectedEvaluationId={trace_id}'
    )

  if print_urls:
    print('🔗 View in MLflow UI:')
    if specific_trace_url:
      print(f'   🎯 This Trace: {specific_trace_url}')
    else:
      print(f'   📊 All Traces: {traces_url}')

  return traces_url, specific_trace_url


def generate_evaluation_links(run_id: str = None):
  """Generate MLflow UI links for viewing evaluation runs."""
  import mlflow

  if mlflow.utils.databricks_utils.is_in_databricks_notebook():
    databricks_host = ensure_https_protocol(mlflow.utils.databricks_utils.get_browser_hostname())
  else:
    databricks_host = ensure_https_protocol(os.getenv('DATABRICKS_HOST'))
  experiment_id = os.getenv('MLFLOW_EXPERIMENT_ID')

  if not databricks_host or not experiment_id:
    print('⚠️ Missing DATABRICKS_HOST or MLFLOW_EXPERIMENT_ID - cannot generate links')
    return

  # General experiment evaluation runs view
  evaluation_runs_url = f'{databricks_host}/ml/experiments/{experiment_id}/evaluation-runs'

  # Specific evaluation run view if run_id provided
  specific_evaluation_url = None
  if run_id:
    specific_evaluation_url = (
      f'{databricks_host}/ml/experiments/{experiment_id}/evaluation-runs?selectedRunUuid={run_id}'
    )

  print('🔗 View in MLflow UI:')

  if specific_evaluation_url:
    print(f'   🎯 This Evaluation Run: {specific_evaluation_url}')
  else:
    print(f'   📊 All Evaluation Runs: {evaluation_runs_url}')


def generate_dataset_link(dataset_id: str = None, print_url: bool = False) -> str:
  """Generate MLflow UI link for viewing a specific evaluation dataset.

  Args:
    dataset_id: The dataset ID to link to
    print_url: Whether to print the URL to stdout

  Returns:
    The dataset URL
  """
  import mlflow

  if mlflow.utils.databricks_utils.is_in_databricks_notebook():
    databricks_host = ensure_https_protocol(mlflow.utils.databricks_utils.get_browser_hostname())
  else:
    databricks_host = ensure_https_protocol(os.getenv('DATABRICKS_HOST'))
  experiment_id = os.getenv('MLFLOW_EXPERIMENT_ID')

  if not databricks_host or not experiment_id:
    if print_url:
      print('⚠️ Missing DATABRICKS_HOST or MLFLOW_EXPERIMENT_ID - cannot generate dataset link')
    return ''

  if dataset_id:
    dataset_url = (
      f'{databricks_host}/ml/experiments/{experiment_id}/datasets?selectedDatasetId={dataset_id}'
    )
  else:
    # General datasets view if no specific ID provided
    dataset_url = f'{databricks_host}/ml/experiments/{experiment_id}/datasets'

  if print_url:
    print('🔗 View evaluation dataset in MLflow UI:')
    print(f'   📊 Dataset: {dataset_url}')

  return dataset_url


def generate_prompt_link(prompt_name: Optional[str] = None, print_url: bool = True) -> str:
  """Generate MLflow UI link for viewing a prompt in the experiment's prompt registry.

  Args:
    prompt_name: The name of the prompt to link to (optional)
    print_url: Whether to print the URL to stdout

  Returns:
    The prompt URL
  """
  import mlflow

  if mlflow.utils.databricks_utils.is_in_databricks_notebook():
    databricks_host = ensure_https_protocol(mlflow.utils.databricks_utils.get_browser_hostname())
  else:
    databricks_host = ensure_https_protocol(os.getenv('DATABRICKS_HOST'))
  experiment_id = os.getenv('MLFLOW_EXPERIMENT_ID')
  uc_catalog = os.getenv('UC_CATALOG')
  uc_schema = os.getenv('UC_SCHEMA')

  if not databricks_host or not experiment_id:
    if print_url:
      print('⚠️ Missing DATABRICKS_HOST or MLFLOW_EXPERIMENT_ID - cannot generate prompt link')
    return ''

  if prompt_name and uc_catalog and uc_schema:
    # Full prompt path with catalog.schema.prompt_name
    full_prompt_name = f'{uc_catalog}.{uc_schema}.{prompt_name}'
    prompt_url = f'{databricks_host}/ml/experiments/{experiment_id}/prompts/{full_prompt_name}'
  elif prompt_name:
    # Just prompt name without catalog/schema
    prompt_url = f'{databricks_host}/ml/experiments/{experiment_id}/prompts/{prompt_name}'
  else:
    # General prompts view if no specific prompt provided
    prompt_url = f'{databricks_host}/ml/experiments/{experiment_id}/prompts'

  if print_url:
    print('🔗 View prompt in MLflow UI, where you can visualize the differences:')
    print(f'   📝 Prompt: {prompt_url}')

  return prompt_url


def generate_evaluation_comparison_link(
  selected_run_id: str, compare_to_run_id: str, print_url: bool = True
) -> str:
  """Generate MLflow UI link for comparing two evaluation runs side-by-side.

  Args:
    selected_run_id: The primary evaluation run ID to view
    compare_to_run_id: The evaluation run ID to compare against
    print_url: Whether to print the URL to stdout

  Returns:
    The comparison URL
  """
  import mlflow

  if mlflow.utils.databricks_utils.is_in_databricks_notebook():
    databricks_host = ensure_https_protocol(mlflow.utils.databricks_utils.get_browser_hostname())
  else:
    databricks_host = ensure_https_protocol(os.getenv('DATABRICKS_HOST'))
  experiment_id = get_mlflow_experiment_id()

  if not databricks_host or not experiment_id:
    if print_url:
      print('⚠️ Missing DATABRICKS_HOST or MLFLOW_EXPERIMENT_ID - cannot generate comparison link')
    return ''

  comparison_url = (
    f'{databricks_host}/ml/experiments/{experiment_id}/evaluation-runs'
    f'?selectedRunUuid={selected_run_id}&compareToRunUuid={compare_to_run_id}'
  )

  if print_url:
    print('🔗 View evaluation comparison in MLflow UI:')
    print(f'   📊 Compare Runs: {comparison_url}')

  return comparison_url


def generate_labeling_schema_link(print_url: bool = True) -> str:
  """Generate MLflow UI link for viewing labeling schemas in the experiment.

  Args:
    print_url: Whether to print the URL to stdout

  Returns:
    The labeling schema URL
  """
  import mlflow

  if mlflow.utils.databricks_utils.is_in_databricks_notebook():
    databricks_host = ensure_https_protocol(mlflow.utils.databricks_utils.get_browser_hostname())
  else:
    databricks_host = ensure_https_protocol(os.getenv('DATABRICKS_HOST'))
  experiment_id = get_mlflow_experiment_id()

  if not databricks_host or not experiment_id:
    if print_url:
      print(
        '⚠️ Missing DATABRICKS_HOST or MLFLOW_EXPERIMENT_ID - cannot generate labeling schema link'
      )
    return ''

  labeling_schema_url = f'{databricks_host}/ml/experiments/{experiment_id}/label-schemas'

  if print_url:
    print('🔗 View labeling schemas in MLflow UI:')
    print(f'   🏷️ Label Schemas: {labeling_schema_url}')

  return labeling_schema_url


def generate_labeling_session_link(session_id: Optional[str] = None, print_url: bool = True) -> str:
  """Generate MLflow UI link for viewing labeling sessions in the experiment.

  Args:
    session_id: The labeling session ID to link to (optional)
    print_url: Whether to print the URL to stdout

  Returns:
    The labeling session URL
  """
  import mlflow

  if mlflow.utils.databricks_utils.is_in_databricks_notebook():
    databricks_host = ensure_https_protocol(mlflow.utils.databricks_utils.get_browser_hostname())
  else:
    databricks_host = ensure_https_protocol(os.getenv('DATABRICKS_HOST'))
  experiment_id = get_mlflow_experiment_id()

  if not databricks_host or not experiment_id:
    if print_url:
      print(
        '⚠️ Missing DATABRICKS_HOST or MLFLOW_EXPERIMENT_ID - cannot generate labeling session link'
      )
    return ''

  if session_id:
    labeling_session_url = (
      f'{databricks_host}/ml/experiments/{experiment_id}/labeling-sessions'
      f'?selectedLabelingSessionId={session_id}'
    )
  else:
    # General labeling sessions view if no specific session ID provided
    labeling_session_url = f'{databricks_host}/ml/experiments/{experiment_id}/labeling-sessions'

  if print_url:
    print('🔗 View labeling sessions in MLflow UI:')
    print(f'   🏷️ Labeling Session: {labeling_session_url}')

  return labeling_session_url
