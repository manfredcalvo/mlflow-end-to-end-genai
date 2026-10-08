# Databricks notebook source
# MAGIC %md
# MAGIC # Generate BGP banking data
# MAGIC
# MAGIC Creates this participant's `<prefix>_bank_*` Delta tables in an existing
# MAGIC `catalog.schema`. Uses Faker with a fixed seed for reproducibility. Driven by
# MAGIC the `catalog`, `schema`, and `prefix` widgets (set by the `bgp_participant_data`
# MAGIC job). If `prefix` is blank it is derived from `current_user()`.

# COMMAND ----------

# MAGIC %pip install faker

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Parameters

# COMMAND ----------

import random
import sys
from datetime import datetime, timedelta
from decimal import Decimal

sys.path.append('../')

from faker import Faker
from pyspark.sql import SparkSession

from mlflow_demo.utils.mlflow_helpers import sanitize_prefix

dbutils.widgets.text("catalog", "workshop_bgp", "UC catalog")
dbutils.widgets.text("schema", "shared_data", "UC schema")
dbutils.widgets.text("prefix", "", "Per-participant prefix (blank = current_user)")


spark = SparkSession.builder.getOrCreate()

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
# Default the prefix to the running user's name so each participant gets their own copy.
PREFIX = dbutils.widgets.get("prefix") or sanitize_prefix(
    spark.sql("select current_user()").collect()[0][0]
)

FAKE = Faker("en_US")
Faker.seed(42)
random.seed(42)

print(f"Generating banking data in {CATALOG}.{SCHEMA} (prefix: '{PREFIX}')")
print(f"  Spark version: {spark.version}")

# COMMAND ----------

# MAGIC %md ## Reference data + row counts

# COMMAND ----------

CITIES = ["Panama City", "Colon", "David", "Santiago", "Chitre", "La Chorrera"]
ACCOUNT_TYPES = ["Checking", "Savings", "Certificate of Deposit"]
CARD_TYPES = ["Visa Platinum", "Mastercard Gold", "Visa Signature", "American Express"]
LOAN_TYPES = ["Mortgage", "Auto", "Personal", "Home Equity"]
TIER_OPTIONS = ["Standard", "Premium", "Private Banking"]
SEGMENTS = ["Retail", "Small Business", "Corporate"]
TXN_CATEGORIES = [
    "Groceries", "Dining", "Gas", "Shopping", "ATM Withdrawal",
    "Transfer", "Salary Deposit", "Bill Payment", "Investment",
    "Insurance", "Fee", "Interest",
]
PRODUCT_CATEGORIES = ["Account", "Loan", "Card", "Investment"]
CURRENCY = "USD"

CUSTOMER_COUNT = 500
ACCOUNT_COUNT = 1200
TXN_COUNT = 10000
LOAN_COUNT = 200
CARD_COUNT = 400
PRODUCT_COUNT = 50
BRANCH_COUNT = 15

# COMMAND ----------

# MAGIC %md ## Generators + table writer

# COMMAND ----------

def generate_customers():
    rows = []
    for i in range(CUSTOMER_COUNT):
        first = FAKE.first_name()
        last = FAKE.last_name()
        cust_id = f"CUST-{i+1:05d}"
        rows.append({
            "customer_id": cust_id,
            "first_name": first,
            "last_name": last,
            "email": f"{first.lower()}.{last.lower()}@email.com",
            "phone": FAKE.phone_number(),
            "customer_since": FAKE.date_between(start_date="-10y", end_date="-1y"),
            "tier": random.choice(TIER_OPTIONS),
            "segment": random.choices(SEGMENTS, weights=[70, 20, 10])[0],
            "country": "Panama",
        })
    return rows


def generate_accounts(customers):
    rows = []
    for i in range(ACCOUNT_COUNT):
        cust = random.choice(customers)
        acct_type = random.choices(ACCOUNT_TYPES, weights=[50, 40, 10])[0]
        balance = Decimal(str(round(random.uniform(50, 50000), 2)))
        rows.append({
            "account_id": f"ACCT-{i+1:06d}",
            "customer_id": cust["customer_id"],
            "account_type": acct_type,
            "balance": balance,
            "currency": CURRENCY,
            "status": random.choices(["Active", "Dormant", "Closed"], weights=[85, 10, 5])[0],
            "opened_date": FAKE.date_between(start_date="-10y", end_date="-1y"),
        })
    return rows


def generate_transactions(accounts):
    rows = []
    end_date = datetime.now()
    for i in range(TXN_COUNT):
        acct = random.choice(accounts)
        if acct["status"] == "Closed":
            continue
        txn_date = end_date - timedelta(days=random.randint(1, 180))
        is_credit = random.random() < 0.35
        base_amount = round(random.uniform(5, 2000), 2)
        category = random.choice(TXN_CATEGORIES)
        if is_credit:
            txn_type = "Credit"
            merchant = random.choice(["Salary", "Transfer In", "Refund", "Interest Payment", "Deposit"])
            amount = base_amount
        else:
            txn_type = "Debit"
            merchant = FAKE.company()
            amount = -base_amount
        rows.append({
            "transaction_id": f"TXN-{i+1:08d}",
            "account_id": acct["account_id"],
            "customer_id": acct["customer_id"],
            "date": txn_date,
            "amount": Decimal(str(amount)),
            "transaction_type": txn_type,
            "merchant": merchant,
            "category": category,
            "description": f"{txn_type}: {merchant} - {category}",
        })
    return rows


def generate_loans(customers):
    rows = []
    for i in range(LOAN_COUNT):
        cust = random.choice(customers)
        loan_type = random.choice(LOAN_TYPES)
        if loan_type == "Mortgage":
            principal = Decimal(str(round(random.uniform(50000, 500000), 2)))
            term = random.choice([180, 240, 360])
            rate = Decimal(str(round(random.uniform(3.5, 7.5), 2)))
        elif loan_type == "Auto":
            principal = Decimal(str(round(random.uniform(15000, 80000), 2)))
            term = random.choice([36, 48, 60, 72])
            rate = Decimal(str(round(random.uniform(4.0, 9.0), 2)))
        else:
            principal = Decimal(str(round(random.uniform(5000, 50000), 2)))
            term = random.choice([12, 24, 36, 60])
            rate = Decimal(str(round(random.uniform(6.0, 15.0), 2)))
        monthly_rate = float(rate) / 100 / 12
        monthly_payment = Decimal(str(round(
            float(principal) * monthly_rate / (1 - (1 + monthly_rate) ** (-term)), 2
        )))
        outstanding = Decimal(str(round(float(principal) * random.uniform(0.2, 0.9), 2)))
        rows.append({
            "loan_id": f"LOAN-{i+1:05d}",
            "customer_id": cust["customer_id"],
            "loan_type": loan_type,
            "principal": principal,
            "interest_rate": rate,
            "term_months": term,
            "outstanding_balance": outstanding,
            "monthly_payment": monthly_payment,
            "status": random.choices(["Active", "Paid Off", "Defaulted"], weights=[80, 15, 5])[0],
        })
    return rows


def generate_credit_cards(customers):
    rows = []
    for i in range(CARD_COUNT):
        cust = random.choice(customers)
        card_type = random.choice(CARD_TYPES)
        if "Platinum" in card_type or "Signature" in card_type:
            credit_limit = Decimal(str(random.choice([10000, 15000, 25000, 50000])))
        else:
            credit_limit = Decimal(str(random.choice([2500, 5000, 7500, 10000])))
        current_balance = Decimal(str(round(float(credit_limit) * random.uniform(0, 0.8), 2)))
        available = credit_limit - current_balance
        rows.append({
            "card_id": f"CARD-{i+1:05d}",
            "customer_id": cust["customer_id"],
            "card_type": card_type,
            "credit_limit": credit_limit,
            "current_balance": current_balance,
            "available_credit": available,
            "status": random.choices(["Active", "Blocked", "Expired"], weights=[90, 5, 5])[0],
        })
    return rows


def generate_products():
    rows = []
    product_defs = [
        ("Premium Checking", "Account", Decimal("0.01"), Decimal("0")),
        ("Basic Savings", "Account", Decimal("0.5"), Decimal("100")),
        ("High-Yield Savings", "Account", Decimal("4.5"), Decimal("1000")),
        ("Certificate of Deposit 12mo", "Account", Decimal("5.0"), Decimal("5000")),
        ("Certificate of Deposit 24mo", "Account", Decimal("5.5"), Decimal("5000")),
        ("Mortgage Loan", "Loan", Decimal("4.5"), Decimal("0")),
        ("Auto Loan", "Loan", Decimal("5.5"), Decimal("0")),
        ("Personal Loan", "Loan", Decimal("8.5"), Decimal("0")),
        ("Home Equity Line", "Loan", Decimal("6.5"), Decimal("0")),
        ("Visa Platinum Card", "Card", Decimal("0"), Decimal("0")),
        ("Mastercard Gold", "Card", Decimal("0"), Decimal("0")),
        ("Investment Fund Conservative", "Investment", Decimal("3.0"), Decimal("10000")),
        ("Investment Fund Moderate", "Investment", Decimal("5.0"), Decimal("10000")),
        ("Investment Fund Aggressive", "Investment", Decimal("7.0"), Decimal("25000")),
    ]
    for i, (name, cat, rate, min_bal) in enumerate(product_defs):
        rows.append({
            "product_id": f"PROD-{i+1:03d}",
            "product_name": name,
            "category": cat,
            "interest_rate": rate,
            "min_balance": min_bal,
            "features": f"Benefits: {', '.join(FAKE.words(3))}",
            "eligibility": "BGP customer with valid ID and minimum balance",
        })
    # Pad to PRODUCT_COUNT
    while len(rows) < PRODUCT_COUNT:
        i = len(rows)
        cat = random.choice(PRODUCT_CATEGORIES)
        rows.append({
            "product_id": f"PROD-{i+1:03d}",
            "product_name": f"{FAKE.bs().title()} {cat}",
            "category": cat,
            "interest_rate": Decimal(str(round(random.uniform(0, 8), 2))),
            "min_balance": Decimal(str(random.choice([0, 100, 1000, 5000, 10000]))),
            "features": f"Features: {', '.join(FAKE.words(4))}",
            "eligibility": "BGP customer with valid ID",
        })
    return rows[:PRODUCT_COUNT]


def generate_branches():
    rows = []
    for i in range(BRANCH_COUNT):
        city = random.choice(CITIES)
        rows.append({
            "branch_id": f"BR-{i+1:03d}",
            "branch_name": f"BGP {city} Branch {i+1}",
            "address": FAKE.street_address(),
            "city": city,
            "phone": FAKE.phone_number(),
            "hours": "Mon-Fri 9:00-17:00, Sat 9:00-13:00",
            "services": "Personal Banking, Business Banking, Mortgages, Investments, ATMs",
        })
    return rows


def write_table(rows, table_name):
    """Write a list of dicts to a (prefixed) Delta table via Spark."""
    if PREFIX:
        table_name = f"{PREFIX}_{table_name}"
    full_name = f"{CATALOG}.{SCHEMA}.{table_name}"
    df = spark.createDataFrame(rows)
    (df.write
     .format("delta")
     .mode("overwrite")
     .option("overwriteSchema", "true")
     .saveAsTable(full_name))
    count = spark.read.table(full_name).count()
    print(f"  Wrote {full_name} ({count} rows)")

# COMMAND ----------

# MAGIC %md ## Generate the datasets

# COMMAND ----------

customers = generate_customers()
accounts = generate_accounts(customers)
transactions = generate_transactions(accounts)
loans = generate_loans(customers)
cards = generate_credit_cards(customers)
products = generate_products()
branches = generate_branches()
print(f"customers={len(customers)} accounts={len(accounts)} transactions={len(transactions)} "
      f"loans={len(loans)} cards={len(cards)} products={len(products)} branches={len(branches)}")

# COMMAND ----------

# MAGIC %md ## Write the Delta tables

# COMMAND ----------

write_table(customers, "bank_customers")
write_table(accounts, "bank_accounts")
write_table(transactions, "bank_transactions")
write_table(loans, "bank_loans")
write_table(cards, "bank_credit_cards")
write_table(products, "bank_products")
write_table(branches, "bank_branches")
print("\nDone! All banking tables created successfully.")
