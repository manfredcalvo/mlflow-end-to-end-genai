"""Pre-workshop PREFLIGHT for the BGP MLflow GenAI workshop.

The instructor no longer creates the catalog/schema or per-participant data —
a workspace **admin** provisions ONE catalog + ONE schema and grants access, and
each participant then self-provisions their own prefixed objects by running the
`bgp_participant_data` + `bgp_participant_setup` jobs from their clone (see
workshop/README.md).

This script only *verifies* the shared schema is present and writable by the
current user, and prints the admin grant checklist. Run it on a Databricks
cluster (Spark required).

Usage (Databricks notebook):
    %run workshop/instructor_setup.py
Env:
    WORKSHOP_CATALOG (default workshop_bgp), WORKSHOP_SCHEMA (default shared_data)
"""

import os

try:
    from pyspark.sql import SparkSession
except ImportError:
    raise ImportError("Run this on a Databricks cluster with Spark available.")

CATALOG = os.getenv("WORKSHOP_CATALOG", "workshop_bgp")
SCHEMA = os.getenv("WORKSHOP_SCHEMA", "shared_data")


def main():
    spark = SparkSession.builder.getOrCreate()
    full = f"{CATALOG}.{SCHEMA}"
    user = spark.sql("select current_user()").collect()[0][0]

    print("=" * 64)
    print("BGP Workshop — preflight")
    print(f"  user:   {user}")
    print(f"  schema: {full}")
    print("=" * 64)

    ok = True

    # 1. Catalog + schema exist and are reachable.
    try:
        spark.sql(f"DESCRIBE SCHEMA {full}")
        print(f"[OK]   schema {full} exists and is reachable")
    except Exception as e:
        ok = False
        print(f"[FAIL] cannot reach schema {full}: {e}")
        print("       -> ask an admin to create the catalog + schema.")

    # 2. Current user can create a table + function (self-provisioning rights).
    probe = f"{full}._preflight_probe"
    try:
        spark.sql(f"CREATE TABLE IF NOT EXISTS {probe} (x INT)")
        spark.sql(f"DROP TABLE IF EXISTS {probe}")
        print(f"[OK]   can CREATE/DROP TABLE in {full}")
    except Exception as e:
        ok = False
        print(f"[FAIL] cannot CREATE TABLE in {full}: {e}")
        print("       -> ask an admin to GRANT CREATE TABLE, CREATE FUNCTION on the schema.")

    print("\nAdmin checklist (one-time, requires schema ownership/MANAGE):")
    print(f"  GRANT USE CATALOG ON CATALOG {CATALOG} TO `<each participant>`;")
    print(f"  GRANT USE SCHEMA, CREATE TABLE, CREATE FUNCTION ON SCHEMA {full} TO `<each participant>`;")
    print(f"  -- after each participant deploys their app, grant its service principal:")
    print(f"  GRANT USE SCHEMA, EXECUTE, SELECT, MODIFY ON SCHEMA {full} TO `<app service principal>`;")
    print("     (MODIFY lets the app write MLflow *_otel_* trace tables.)")

    print("\n[READY]" if ok else "\n[NOT READY] resolve the FAIL items above.")
    return ok


if __name__ == "__main__":
    main()
