"""function_tool wrappers around the BGP banking UC functions.

Uses DatabricksFunctionClient with serverless execution, so we don't need a
SQL warehouse grant for the app SP — UC routes the function call to managed
serverless compute. The app SP only needs USE SCHEMA + EXECUTE on the schema.

The target schema is read from UC_TOOL_SCHEMA (set by the DAB bundle), e.g.
`casaman_aws_stable_catalog.bgp_workshop`. This keeps the banking chat agent
fully parameterized from the bundle variables.
"""

import asyncio
import logging
import os
from typing import Optional

from agents import function_tool
from databricks_openai import DatabricksFunctionClient

logger = logging.getLogger(__name__)

# Banking UC functions live in UC_TOOL_SCHEMA (catalog.schema). Fall back to
# TELCO_FUNCTION_PREFIX for backward compatibility.
TARGET_FN_PREFIX = os.getenv('UC_TOOL_SCHEMA') or os.getenv('TELCO_FUNCTION_PREFIX', '')
# UC_TOOL_PREFIX namespaces the UC functions per participant (e.g. jane_doe_get_...),
# matching the <prefix>_get_* functions the data job creates in the shared schema.
# The LLM-facing tool names stay unprefixed; only the resolved UC name carries it.
_FN_NAME_PREFIX = (lambda x: f'{x}_' if x else '')(os.getenv('UC_TOOL_PREFIX', ''))

# Lazy-initialized singleton: DatabricksFunctionClient bootstraps a
# WorkspaceClient + serverless session, both of which we want to share across
# tool calls. Initializing per-call would re-auth on every tool invocation.
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


def _execute_sync(fn_name: str, params: dict) -> str:
  """Run the UC function on serverless and return its first-column value."""
  client = _get_client()
  result = client.execute_function(f'{TARGET_FN_PREFIX}.{_FN_NAME_PREFIX}{fn_name}', params)
  if getattr(result, 'error', None):
    return f'{{"error": "{result.error}"}}'
  value = getattr(result, 'value', None)
  return value or '{}'


async def _call_function(fn_name: str, params: dict) -> str:
  """Async wrapper that executes the UC function in a worker thread."""
  return await asyncio.to_thread(_execute_sync, fn_name, params)


# ---------------------------------------------------------------------------
# Tool definitions — names and docstrings drive the LLM's tool selection.
# These wrap the 7 BGP banking UC functions.
# ---------------------------------------------------------------------------


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
  """List BGP banking products, optionally filtered by category.

  Args:
    category: Product category: Account, Loan, Card, or Investment. Omit for all products.
  """
  return await _call_function('get_product_catalog', {'p_category': category})


@function_tool
async def get_branch_info(city: Optional[str] = None) -> str:
  """List BGP branches, optionally filtered by city (e.g., Panama City, Colon,
  David). Returns address, phone, hours, and services.

  Args:
    city: City name to filter by. Omit for all branches.
  """
  return await _call_function('get_branch_info', {'p_city': city})


TELCO_TOOLS = [
  get_customer_profile,
  get_account_balance,
  get_transaction_history,
  get_loan_details,
  get_credit_card_info,
  get_product_catalog,
  get_branch_info,
]
