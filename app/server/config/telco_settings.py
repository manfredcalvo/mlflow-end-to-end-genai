"""Settings for the telco support agent backend.

Read from env vars under the TELCO_* prefix where they could collide with
NFL settings (notably MLFLOW_EXPERIMENT_ID). Auth follows the same pattern
as the original telco app: prefer OAuth M2M (auto-injected by Databricks
Apps) when client id/secret are present, fall back to PAT for local dev.
"""

import logging
import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class TelcoSettings(BaseSettings):
  """Telco support agent settings."""

  model_config = SettingsConfigDict(
    env_file='.env', case_sensitive=False, extra='ignore'
  )

  environment: str = Field(default_factory=lambda: os.getenv('ENV', 'prod'))

  databricks_host: str = Field(
    default_factory=lambda: os.getenv('DATABRICKS_HOST', '')
  )
  databricks_token: str = Field(
    default_factory=lambda: os.getenv('DATABRICKS_TOKEN', '')
  )
  databricks_client_id: str = Field(
    default_factory=lambda: os.getenv('DATABRICKS_CLIENT_ID', '')
  )
  databricks_client_secret: str = Field(
    default_factory=lambda: os.getenv('DATABRICKS_CLIENT_SECRET', '')
  )
  databricks_endpoint_name: str = Field(
    default_factory=lambda: os.getenv(
      'DATABRICKS_ENDPOINT_NAME', 'prod-telco-customer-support-agent'
    )
  )

  mlflow_experiment_path_override: str = Field(
    default_factory=lambda: os.getenv('TELCO_MLFLOW_EXPERIMENT_PATH', '')
  )
  # validation_alias pins this to TELCO_MLFLOW_EXPERIMENT_ID. Without it,
  # pydantic-settings auto-binds the field name to the env var
  # MLFLOW_EXPERIMENT_ID (which the app sets to '' and resolves by name),
  # silently blanking the telco experiment id.
  mlflow_experiment_id: str = Field(
    default='',
    validation_alias='TELCO_MLFLOW_EXPERIMENT_ID',
  )

  request_timeout: int = Field(default=300)
  max_retries: int = Field(default=3)

  def __init__(self, **kwargs):
    """Normalize host and log auth state."""
    super().__init__(**kwargs)

    if self.databricks_host and not self.databricks_host.startswith(
      ('http://', 'https://')
    ):
      self.databricks_host = f'https://{self.databricks_host}'

    logger.info(
      'Telco settings: env=%s, host=%s, endpoint=%s, auth=%s, mlflow_path=%s',
      self.environment,
      self.databricks_host,
      self.databricks_endpoint_name,
      self.auth_method,
      self.mlflow_experiment_path,
    )

    if self.auth_method == 'none':
      logger.warning(
        'No Databricks auth for telco service - chat will return error responses'
      )

  def get_demo_customer_ids(self) -> list[str]:
    """Read demo customer IDs from env or fall back to defaults."""
    customers_str = os.getenv(
      'DEMO_CUSTOMER_IDS',
      'CUST-00003,CUST-00005,CUST-00011,CUST-00015,CUST-00020,CUST-00028,CUST-00040,CUST-00041,CUST-00046,CUST-00049',
    )
    return [c.strip() for c in customers_str.split(',') if c.strip()]

  @property
  def demo_customer_ids(self) -> list[str]:
    """Get demo customer IDs."""
    return self.get_demo_customer_ids()

  @property
  def databricks_endpoint(self) -> str:
    """Full Databricks model-serving invocations URL."""
    host = self.databricks_host.rstrip('/')
    return f'{host}/serving-endpoints/{self.databricks_endpoint_name}/invocations'

  @property
  def auth_method(self) -> str:
    """Auth method in priority order: oauth > token > none."""
    if self.databricks_client_id and self.databricks_client_secret:
      return 'oauth'
    if self.databricks_token:
      return 'token'
    return 'none'

  @property
  def has_auth(self) -> bool:
    """Whether any auth is configured."""
    return self.auth_method != 'none'

  @property
  def mlflow_experiment_path(self) -> str:
    """MLflow experiment path for telco traces (separate from NFL)."""
    if self.mlflow_experiment_path_override:
      return self.mlflow_experiment_path_override
    env = self.environment
    return f'/Shared/telco_support_agent/{env}/{env}_telco_support_agent'


@lru_cache
def get_telco_settings() -> TelcoSettings:
  """Cached settings instance."""
  return TelcoSettings()
