"""MLflow evaluation logic for the Banking Customer Support Assistant."""

import os

import mlflow
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import Guidelines, Safety, RelevanceToQuery

from mlflow_demo.utils.mlflow_helpers import generate_evaluation_links

# Prompt registry configuration
DEV_PROMPT_ALIAS = 'development'


def validate_env_vars():
  """Validate required environment variables are set."""
  PROMPT_NAME = os.getenv('PROMPT_NAME')
  PROMPT_ALIAS = os.getenv('PROMPT_ALIAS')
  if not PROMPT_NAME or not PROMPT_ALIAS:
    raise Exception('PROMPT_NAME and PROMPT_ALIAS environment variables must be set')
  return PROMPT_NAME, PROMPT_ALIAS


REGRESSION_DATASET_NAME = 'regression_set'
FIX_DATASET_NAME = 'low_accuracy'

UC_CATALOG = os.environ.get('UC_CATALOG')
UC_SCHEMA = os.environ.get('UC_SCHEMA')

# --- Built-in Scorers ---

# Tone of voice Guideline - Ensure professional banking tone
tone = Guidelines(
  name='tone',
  guidelines='The response maintains a professional, knowledgeable banking tone appropriate for a customer support assistant at the bank.',
)

# Built-in safety scorer - checks for harmful content
safety = Safety()

# Built-in relevance scorer - checks if response addresses the user's request
relevance_to_query = RelevanceToQuery()

# --- Custom Judges via make_judge() ---

accuracy = make_judge(
  name='accuracy',
  instructions="""Evaluate whether the agent's response correctly references factual information from the execution trace.

  Analyze the full execution {{ trace }} to find tool call results, then assess the response against these rules:
  - All account numbers, balances, and transaction amounts must match the data returned by tools
  - Customer names, account types, and product names must be accurate
  - Transaction details (dates, merchants, amounts) must match tool output
  - Loan details (principal, interest rate, outstanding balance) must be accurate
  - Credit card info (limits, balances, available credit) must match tool results
  - No fabricated financial data or invented account details
  - If tools returned no data, the response should acknowledge the lack of data rather than guess
  - It is acceptable to provide general banking knowledge as context, but specific claims must be data-backed

  Respond with 'yes' if the response is factually accurate, or 'no' if it contains errors or fabrications.""",
)

relevance = make_judge(
  name='relevance',
  instructions="""Evaluate whether the agent's response directly addresses the user's question.

  User's request: {{ inputs }}
  Agent's response: {{ outputs }}

  Assess based on these rules:
  - The response focuses on the specific banking question asked (accounts, transactions, loans, cards, products)
  - Recommendations and information are relevant to the customer's banking needs
  - The response does not go off-topic with unrelated banking analysis
  - If the question asks about a specific account or transaction, the answer addresses that specific item
  - Financial breakdowns should be relevant to the question context

  Respond with 'yes' if the response is relevant, or 'no' if it is off-topic or misses the question.""",
)

actionability = make_judge(
  name='actionability',
  instructions="""Evaluate whether the agent's response provides actionable guidance for the customer.

  Agent's response: {{ outputs }}

  Assess based on these rules:
  - Includes specific next steps or guidance (e.g., how to dispute a transaction, how to apply for a product)
  - Recommendations are practical and implementable for a banking customer
  - Addresses the customer's specific situation with relevant suggestions
  - Avoids vague advice like 'contact support' without specifics when data is available
  - When data supports it, suggests specific products or services that match the customer's profile

  Respond with 'yes' if the response is actionable, or 'no' if it is vague or unhelpful.""",
)

response_is_grounded = make_judge(
  name='response_is_grounded',
  instructions="""Evaluate whether the agent's response is grounded in the data retrieved by tool calls.

  Analyze the full execution {{ trace }} to find tool call results, then assess whether:
  - The response only makes claims supported by tool output data
  - Account balances, transaction amounts, and financial figures cited appear in the tool results
  - The response does not hallucinate data that was not returned by any tool
  - If no tool data was retrieved, the response acknowledges the lack of data

  Respond with 'yes' if the response is grounded in tool results, or 'no' if it contains hallucinated or unsupported claims.""",
)

# Convenience list of all scorers for easy use in evaluation
SCORERS = [tone, safety, relevance_to_query, accuracy, relevance, actionability, response_is_grounded]


def get_scorers(prefix=''):
  """Return namespaced scorers for production monitoring.

  When multiple workshop participants share a workspace, scorer names must be
  namespaced (e.g., 'p01_accuracy') to avoid collisions. For offline evaluation,
  scorer names are per-experiment and do not need namespacing.
  """
  p = f'{prefix}_' if prefix else ''

  _tone = Guidelines(
    name=f'{p}tone',
    guidelines='The response maintains a professional, knowledgeable banking tone appropriate for a customer support assistant at the bank.',
  )
  _safety = Safety()
  if p:
    _safety = Safety(name=f'{p}safety')
  _relevance_to_query = RelevanceToQuery()
  if p:
    _relevance_to_query = RelevanceToQuery(name=f'{p}relevance_to_query')

  _accuracy = make_judge(
    name=f'{p}accuracy',
    instructions=accuracy.instructions,
  )
  _relevance = make_judge(
    name=f'{p}relevance',
    instructions=relevance.instructions,
  )
  _actionability = make_judge(
    name=f'{p}actionability',
    instructions=actionability.instructions,
  )
  _response_is_grounded = make_judge(
    name=f'{p}response_is_grounded',
    instructions=response_is_grounded.instructions,
  )

  return [_tone, _safety, _relevance_to_query, _accuracy, _relevance, _actionability, _response_is_grounded]


def run_evaluation():
  """Run evaluation on recent traces."""
  print('\n🔍 Loading recent traces from the DC assistant...')

  # Load recent traces for evaluation
  traces = mlflow.search_traces(
    max_results=3,
    filter_string='status = "OK"',
    order_by=['timestamp DESC'],
  )
  print(f'✅ Found {len(traces)} traces for evaluation')

  # Now, let's run evaluation using this scorer
  eval_results = mlflow.genai.evaluate(data=traces, scorers=SCORERS)

  print('\n📊 Evaluation completed!')
  print(f'🆔 Run ID: {eval_results.run_id}')

  # Generate and display evaluation links
  generate_evaluation_links(eval_results.run_id)

  return eval_results
