# Databricks notebook source
# MAGIC %md
# MAGIC # Create banking UC functions
# MAGIC
# MAGIC Creates this participant's 7 `<prefix>_get_*` SQL UC functions, each reading
# MAGIC the matching `<prefix>_bank_*` table. Run after `generate_banking_data`.
# MAGIC Driven by the `catalog`, `schema`, and `prefix` widgets (set by the
# MAGIC `participant_data` job); blank `prefix` is derived from `current_user()`.

# COMMAND ----------

# MAGIC %md ## Parameters

# COMMAND ----------

import sys

sys.path.append('../')

from pyspark.sql import SparkSession

from mlflow_demo.utils.mlflow_helpers import sanitize_prefix

dbutils.widgets.text("catalog", "workshop_bank", "UC catalog")
dbutils.widgets.text("schema", "shared_data", "UC schema")
dbutils.widgets.text("prefix", "", "Per-participant prefix (blank = current_user)")
dbutils.widgets.text("app_name", "", "Databricks App name (its service principal gets EXECUTE on the functions)")


spark = SparkSession.builder.getOrCreate()

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
# Default the prefix to the running user's name (matches generate_banking_data).
PREFIX = dbutils.widgets.get("prefix") or sanitize_prefix(
    spark.sql("select current_user()").collect()[0][0]
)
p = f"{PREFIX}_" if PREFIX else ""
full_schema = f"{CATALOG}.{SCHEMA}"
print(f"Creating 7 UC functions in {full_schema} (prefix: '{PREFIX}')")

# COMMAND ----------

# MAGIC %md ## Function definitions

# COMMAND ----------

FUNCTIONS_SQL = [
    # 1. Customer profile
    """
    CREATE OR REPLACE FUNCTION {full_schema}.get_customer_profile(
      p_customer_id STRING COMMENT 'Customer ID (e.g., CUST-00001)'
    )
    RETURNS STRING
    LANGUAGE SQL
    COMMENT 'JSON: customer profile including name, email, tier, segment, and customer since date'
    RETURN (
      SELECT to_json(
        named_struct(
          'customer_id', customer_id,
          'first_name', first_name,
          'last_name', last_name,
          'email', email,
          'phone', phone,
          'customer_since', customer_since,
          'tier', tier,
          'segment', segment,
          'country', country
        )
      )
      FROM bank_customers
      WHERE customer_id = p_customer_id
      LIMIT 1
    )
    """,

    # 2. Account balance
    """
    CREATE OR REPLACE FUNCTION {full_schema}.get_account_balance(
      p_customer_id STRING COMMENT 'Customer ID'
    )
    RETURNS STRING
    LANGUAGE SQL
    COMMENT 'JSON array: all accounts and balances for a customer'
    RETURN (
      SELECT to_json(
        collect_list(
          named_struct(
            'account_id', account_id,
            'account_type', account_type,
            'balance', balance,
            'currency', currency,
            'status', status,
            'opened_date', opened_date
          )
        )
      )
      FROM (
        SELECT account_id, account_type, balance, currency, status, opened_date
        FROM bank_accounts
        WHERE customer_id = p_customer_id AND status = 'Active'
        ORDER BY balance DESC
        LIMIT 20
      )
    )
    """,

    # 3. Transaction history
    """
    CREATE OR REPLACE FUNCTION {full_schema}.get_transaction_history(
      p_customer_id STRING COMMENT 'Customer ID',
      p_days_back INT COMMENT 'Number of days to look back (e.g., 30 for last 30 days)'
    )
    RETURNS STRING
    LANGUAGE SQL
    COMMENT 'JSON array: transactions for the last N days for a customer'
    RETURN (
      SELECT to_json(
        collect_list(
          named_struct(
            'transaction_id', transaction_id,
            'account_id', account_id,
            'date', date,
            'amount', amount,
            'transaction_type', transaction_type,
            'merchant', merchant,
            'category', category,
            'description', description
          )
        )
      )
      FROM (
        SELECT transaction_id, account_id, date, amount, transaction_type, merchant, category, description
        FROM bank_transactions
        WHERE customer_id = p_customer_id
          AND date >= date_sub(current_date(), COALESCE(p_days_back, 90))
        ORDER BY date DESC
        LIMIT 100
      )
    )
    """,

    # 4. Loan details
    """
    CREATE OR REPLACE FUNCTION {full_schema}.get_loan_details(
      p_customer_id STRING COMMENT 'Customer ID'
    )
    RETURNS STRING
    LANGUAGE SQL
    COMMENT 'JSON array: all loan details for a customer'
    RETURN (
      SELECT to_json(
        collect_list(
          named_struct(
            'loan_id', loan_id,
            'loan_type', loan_type,
            'principal', principal,
            'interest_rate', interest_rate,
            'term_months', term_months,
            'outstanding_balance', outstanding_balance,
            'monthly_payment', monthly_payment,
            'status', status
          )
        )
      )
      FROM (
        SELECT loan_id, loan_type, principal, interest_rate, term_months, outstanding_balance, monthly_payment, status
        FROM bank_loans
        WHERE customer_id = p_customer_id
        ORDER BY outstanding_balance DESC
        LIMIT 20
      )
    )
    """,

    # 5. Credit card info
    """
    CREATE OR REPLACE FUNCTION {full_schema}.get_credit_card_info(
      p_customer_id STRING COMMENT 'Customer ID'
    )
    RETURNS STRING
    LANGUAGE SQL
    COMMENT 'JSON array: credit card info (limits, balances, available credit) for a customer'
    RETURN (
      SELECT to_json(
        collect_list(
          named_struct(
            'card_id', card_id,
            'card_type', card_type,
            'credit_limit', credit_limit,
            'current_balance', current_balance,
            'available_credit', available_credit,
            'status', status
          )
        )
      )
      FROM (
        SELECT card_id, card_type, credit_limit, current_balance, available_credit, status
        FROM bank_credit_cards
        WHERE customer_id = p_customer_id AND status = 'Active'
        LIMIT 10
      )
    )
    """,

    # 6. Product catalog
    """
    CREATE OR REPLACE FUNCTION {full_schema}.get_product_catalog(
      p_category STRING COMMENT 'Product category filter: Account, Loan, Card, Investment. NULL returns all.'
    )
    RETURNS STRING
    LANGUAGE SQL
    COMMENT 'JSON array: banking products filtered by category (NULL = all products)'
    RETURN (
      SELECT to_json(
        collect_list(
          named_struct(
            'product_id', product_id,
            'product_name', product_name,
            'category', category,
            'interest_rate', interest_rate,
            'min_balance', min_balance,
            'features', features,
            'eligibility', eligibility
          )
        )
      )
      FROM (
        SELECT product_id, product_name, category, interest_rate, min_balance, features, eligibility
        FROM bank_products
        WHERE p_category IS NULL OR category = p_category
        ORDER BY interest_rate DESC
        LIMIT 50
      )
    )
    """,

    # 7. Branch info
    """
    CREATE OR REPLACE FUNCTION {full_schema}.get_branch_info(
      p_city STRING COMMENT 'City name (e.g., Springfield, Fairview, Riverton). NULL returns all branches.'
    )
    RETURNS STRING
    LANGUAGE SQL
    COMMENT 'JSON array: branch info filtered by city (NULL = all branches)'
    RETURN (
      SELECT to_json(
        collect_list(
          named_struct(
            'branch_id', branch_id,
            'branch_name', branch_name,
            'address', address,
            'city', city,
            'phone', phone,
            'hours', hours,
            'services', services
          )
        )
      )
      FROM (
        SELECT branch_id, branch_name, address, city, phone, hours, services
        FROM bank_branches
        WHERE p_city IS NULL OR city = p_city
        LIMIT 20
      )
    )
    """,
]

# COMMAND ----------

# MAGIC %md ## Create the functions

# COMMAND ----------

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

for i, sql_template in enumerate(FUNCTIONS_SQL, 1):
    sql = sql_template.format(full_schema=full_schema)
    # Prefix the function name and fully-qualify + prefix the table it reads, so each
    # participant's functions bind to their own <prefix>_bank_* tables in the shared schema.
    sql = sql.replace(f"FUNCTION {full_schema}.get_", f"FUNCTION {full_schema}.{p}get_")
    sql = sql.replace("FROM bank_", f"FROM {full_schema}.{p}bank_")
    func_name = sql.split("FUNCTION ")[1].split("(")[0].strip()
    spark.sql(sql)
    print(f"  [{i}/7] Created function: {func_name}")

# COMMAND ----------

# MAGIC %md ## Verify

# COMMAND ----------

result = spark.sql(f"SELECT {full_schema}.{p}get_customer_profile('CUST-00001')").collect()
if result and result[0][0]:
    print(f"  {p}get_customer_profile('CUST-00001') returned data: {result[0][0][:100]}...")
else:
    print("  WARNING: get_customer_profile returned no data")

result = spark.sql(f"SELECT {full_schema}.{p}get_branch_info('Springfield')").collect()
if result and result[0][0]:
    print(f"  {p}get_branch_info('Springfield') returned data")

# COMMAND ----------

# MAGIC %md ## Grant EXECUTE to the app's service principal
# MAGIC
# MAGIC The functions can't be DAB app resources (they don't exist until this job
# MAGIC runs, and the app is created by the same deploy that creates this job) —
# MAGIC so this job, running as the participant who OWNS the functions, grants the
# MAGIC app's service principal EXECUTE right after creating them. Idempotent.

# COMMAND ----------

app_name = dbutils.widgets.get("app_name")
if app_name:
    from databricks.sdk import WorkspaceClient

    SHORT_NAMES = [
        "get_customer_profile", "get_account_balance", "get_transaction_history",
        "get_loan_details", "get_credit_card_info", "get_product_catalog", "get_branch_info",
    ]
    sp = WorkspaceClient().apps.get(name=app_name).service_principal_client_id
    print(f"Granting EXECUTE on the {p}get_* functions to app SP {sp} ({app_name})...")
    for i, short in enumerate(SHORT_NAMES, 1):
        func = f"{full_schema}.{p}{short}"
        spark.sql(f"GRANT EXECUTE ON FUNCTION {func} TO `{sp}`")
        print(f"  [{i}/7] GRANT EXECUTE ON FUNCTION {func}")
else:
    print("No app_name widget set — skipping the app-SP EXECUTE grants (set app_name when running the job).")

print("\nDone! All 7 UC functions created successfully.")
