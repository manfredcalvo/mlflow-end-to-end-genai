# BGP Bank — MLflow 3 GenAI Workshop

A hands-on **MLflow 3 GenAI** workshop built around a **banking customer-support agent** for BGP Bank (Panama). Participants deploy their own agent app, use it to generate traces and feedback, then work through notebooks that cover the full quality loop — observe, evaluate, find & fix issues, human review, monitor, and optimize — before redeploying the improved agent.

The agent is a tool-calling assistant that answers questions about a customer's **accounts, transactions, loans, credit cards, products, and branches** by calling Unity Catalog functions. Every turn is traced end-to-end by MLflow and stored in Unity Catalog.

**Learn more about MLflow 3:**

- [Agent observability & quality docs](https://docs.databricks.com/aws/en/mlflow3/genai/)
- [Blog: MLflow 3.0](https://www.databricks.com/blog/mlflow-30-unified-ai-experimentation-observability-and-governance)

## Requirements

**To deploy and take the workshop** (participants): just these —
- **Databricks CLI v1.19.0 or newer** — required to deploy. The agent's LLM is served through an AI Gateway **model service** (`resources/bgp_model_service.yml`), a Beta Databricks Asset Bundles resource (`resources.model_services`) that **older CLIs silently ignore** — `bundle deploy` would create nothing and report no error. Check with `databricks --version`; upgrade with `brew upgrade databricks` or the [install script](https://docs.databricks.com/aws/en/dev-tools/cli/install.html). Confirm support with `databricks bundle schema | grep model_services`.
- **Git** (to clone the repo).
- **An authenticated CLI profile** — `databricks auth login --host <workspace-url> --profile <name>`; pass it to every command with `--profile`.

No local Python, uv, or Node needed — the built frontend is committed (`app/client/build/`), and all setup runs as serverless jobs in the workspace.

**To develop the frontend/app locally** (maintainers only): Python 3.13+ with [uv](https://docs.astral.sh/uv/), and Node.js 18+ / npm.

**Workspace**: Unity Catalog, Model Serving + Mosaic AI Gateway, Databricks Apps, and a SQL warehouse (MLflow trace storage + eval-dataset reads; its ID is the `warehouse_id` default in `databricks.yml`).

## Project structure

```
├── app/                  # The deployable Databricks App
│   ├── server/           #   FastAPI backend (telco banking chat + feedback + tracing)
│   └── client/           #   React frontend (Vite + shadcn/ui); build/ is committed
├── notebooks/            # Workshop labs 0–6
├── mlflow_demo/          # Shared Python library imported by the app AND the notebooks
│   ├── agent/            #   Agent, prompts, config
│   ├── evaluation/       #   Scorers / LLM judges
│   └── utils/            #   MLflow helpers (incl. the notebook env setup)
├── workshop/            # Self-service job notebooks (banking data + UC functions)
├── resources/            # DAB resource definitions (app, experiment, model service, jobs)
└── databricks.yml        # Databricks Asset Bundle (variables, targets, sync)
```

### Notebooks (`notebooks/`)

| Notebook | Topic |
|----------|-------|
| `0_demo_overview` | Introduction and setup |
| `1_observe_with_traces` | MLflow tracing for agent observability |
| `2_create_quality_metrics` | Building LLM judges and running evaluations |
| `3_find_fix_quality_issues` | Identifying and fixing quality issues from eval results |
| `4_human_review` | Expert labeling sessions for ground truth |
| `5_production_monitoring` | Scheduled evaluation monitoring in production |
| `6_prompt_optimization` | Automatically improving the prompt with GEPA |

## Workshop flow

| Phase | Activity |
|-------|----------|
| 0. Pre-workshop (admin) | Create the shared catalog + schema; grant participants CREATE rights (see `workshop/README.md`) |
| 1. Deploy (participant) | 4 self-service CLI commands — everything derives from their username |
| 2. Use & feedback | Chat with the app, submit thumbs up/down, explore traces in MLflow |
| 3. Notebook labs | Work through notebooks 0–6 in the workspace (nb 1 generates the sample data, nb 2 attaches judge assessments) |
| 4. Optimize & redeploy | Run GEPA, register the improved prompt, `bundle run bgp_agent` |

## Data & tools

Synthetic banking data is generated with Faker into 7 Delta tables (`bank_customers`, `bank_accounts`, `bank_transactions`, `bank_loans`, `bank_credit_cards`, `bank_products`, `bank_branches`) and exposed through 7 Unity Catalog SQL functions (`get_customer_profile`, `get_account_balance`, `get_transaction_history`, `get_loan_details`, `get_credit_card_info`, `get_product_catalog`, `get_branch_info`).

**Per-participant provisioning:** every participant shares **one admin-provisioned catalog + schema** and creates their *own* copy, namespaced by a username prefix (`<prefix>_bank_*` tables, `<prefix>_get_*` functions, where `jane.doe@… → jane_doe`). Each participant runs the serverless DAB job **`bgp_participant_data`** (`resources/bgp_banking_data.yml`, scripts in `workshop/`) — it assumes the catalog/schema already exist and never creates them. The agent resolves the prefix at runtime via the `UC_TOOL_PREFIX` env var, so the LLM-facing tool names stay unprefixed. See `workshop/README.md` for the admin grant checklist.

## Deployment (Databricks Asset Bundles)

Deployment is 100% DAB-driven — no shell `sed`, no hand-edited `app.yaml`, no local builds (the frontend `build/` is committed; `requirements.txt` is the committed source of truth). Each participant self-deploys from their clone — every per-participant value (object prefix, prompt name, model service, app name, experiment) is **derived from their username**:

```bash
databricks bundle deploy --target dev --profile <p>                    # app config, experiment, model service, data job
databricks bundle run bgp_participant_data --target dev --profile <p>  # <prefix>_bank_* tables + <prefix>_get_* functions
databricks bundle deploy --target dev --profile <p>                    # 2nd deploy: the Genie space (tables now exist)
databricks bundle run bgp_agent --target dev --profile <p>             # deploy + start the app
```

Zero `--var` flags needed (override `--var app_name=...` only for long/underscored usernames). The **first deploy reports an expected error on the Genie space** — Genie validates that its tables exist, and the bank tables come from the data job; the other resources deploy fine, and the second deploy (after the data job) creates the space. The prompt registers itself on the agent's first import (notebook 1), and the sample traces + judge assessments are generated by notebooks 1–2 as part of the labs.

**CI/CD deploys the shared prod app on push to `main`** via `.github/workflows/deploy.yml` — or the equivalent CircleCI example in `.circleci/config.yml` (same steps, same variables: `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `SQL_WAREHOUSE_ID`, `NOTEBOOK_ROOT`), showing the bundle is CI-provider-agnostic.

**Example Genie agent** (`resources/bgp_genie_space.yml`): a Genie space over the participant's `<prefix>_bank_*` tables — natural-language banking analytics with guided instructions and example SQL, created by the second deploy and usable in the Genie UI.

### How the app gets its config

The app's command and environment live in the DAB app resource (`resources/bgp_agent_app.yml`), not a static `app.yaml`. Notably, the MLflow experiment id/name are sourced from the **experiment resource reference** (`${resources.experiments.bgp_workshop_experiment.id}` / `.name`), so they are never hardcoded and always track the experiment the bundle created.

### Service-principal UC access (fully declarative)

The app's service principal needs **no manual grants** — its UC access is split between the DAB app resources and the self-service data job:

- **App resources** (`resources/bgp_agent_app.yml`, attached at deploy): the SQL warehouse, the MLflow experiment (CAN_EDIT), and the MLflow **trace tables** — named `<prefix>_traces_otel_*` (MODIFY; spans also SELECT) and the `<prefix>_traces_trace_metadata`/`_trace_unified` views (SELECT). The experiment resource sets `uc_trace_location.table_prefix = <participant_prefix>_traces`, so every trace-table name derives purely from the bundle's `participant_prefix` — no workspace-id or experiment-id is embedded anywhere. The DAB creates the experiment and its tables *before* the app within the same deploy. Attaching any UC securable also auto-grants the parent `USE CATALOG`/`USE SCHEMA`.
- **The data job** (`bgp_participant_data` → `workshop/create_uc_functions.py`): the 7 `<prefix>_get_*` functions can't be app resources (they don't exist until the job runs — and the job's definition is created by the same deploy), so the job — running as the participant, who **owns** their functions — grants the app SP `EXECUTE` right after creating them. The functions are DEFINER-rights, so no bank-table SELECT is needed.

Everything fits the standard 20-resource app cap (9 used).

## Local development (maintainers only)

```bash
# Backend (FastAPI on :8000) — needs a Python 3.13+ env, e.g. uv sync && source .venv/bin/activate
cd app && uvicorn server.app:app --reload

# Frontend (Vite on :3000) — in a separate terminal
cd app/client && npm run dev
```

## Optional: Unity Gateway CLI (`ug`) for coding agents

This repo's agent LLM already runs through a [Unity Gateway model service](https://docs.databricks.com/aws/en/ai-gateway/model-services). The `ug` CLI extends the same governed access to your **coding agents** (Claude Code, Codex, Gemini CLI, and others), launching them through your Databricks workspace. Entirely optional — useful for working on this repo with a governed, workspace-backed agent. See the [ug CLI docs](https://docs.databricks.com/aws/en/ai-gateway/coding-agent-ug-cli) for the full reference.

```bash
# Prerequisites: Python 3.12+ and uv (curl -LsSf https://astral.sh/uv/install.sh | sh)

# 1. Install the Unity Gateway CLI
uv tool install git+https://github.com/databricks/unity-gateway
ug --version

# 2. Configure (OAuth sign-in to your workspace, guided)
ug configure
#   ...or specify the workspace up front:
ug configure --workspace https://<your-workspace>.cloud.databricks.com
#   ...or use a personal access token from an existing CLI profile:
databricks configure --host https://<your-workspace>.cloud.databricks.com   # once, saves the profile
ug configure --profile <profile-name> --use-pat

# 3. Sanity-check the installation/configuration
ug doctor

# 4. Launch your agent through the gateway
ug claude          # Claude Code
# ug codex         # Codex        # ug gemini    # Gemini CLI
# ug opencode      # OpenCode     # ug copilot   # GitHub Copilot CLI
# ug cursor        # Cursor Agent # ug pi        # Pi
# ug               # your admin's default agent
```

Notes: `ug claude --refresh` re-pulls workspace auth/models before launching; `ug upgrade` (or `uv tool upgrade unity-gateway`) updates the CLI; `ug` backs up any agent config it replaces and `ug revert` restores it.

Bundle variables (`databricks.yml`): `app_name` (defaults to `bgp-agent-${participant_prefix}`), `uc_catalog`, `uc_schema` (the single shared schema; defaults `workshop_bgp`/`shared_data`), `participant_prefix` (defaults to the deploying user's short name), `warehouse_id` (defaults to the shared workshop warehouse), `llm_foundation_model` (defaults to Claude Sonnet 5), `demo_customer_ids`, `notebook_root`. Everything else is derived: tool schema = `${uc_catalog}.${uc_schema}`, model service = `<prefix>_bgp_agent_llm`, prompt = `<prefix>_bgp_support_prompt`. Targets: `dev` (default, development mode) and `prod` (production mode, used by CI/CD).

Runtime app env (set from the DAB config block): `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_ID`/`_NAME` (from the experiment resource), `MLFLOW_TRACING_SQL_WAREHOUSE_ID`, `SQL_WAREHOUSE_ID`, `UC_CATALOG`, `UC_SCHEMA`, `UC_TOOL_SCHEMA`, `UC_TOOL_PREFIX` (per-participant namespace), `PROMPT_NAME` (`<prefix>_bgp_support_prompt`), `SCORER_PREFIX`, `LLM_MODEL_SERVICE` (the AI Gateway model service `<catalog>.<schema>.<prefix>_bgp_agent_llm`), `DEMO_CUSTOMER_IDS`, `TELCO_MLFLOW_EXPERIMENT_ID`/`_PATH`.

## MLflow 3 capabilities demonstrated

- **GenAI observability** — tracing of agent tool calls and LLM responses, stored in Unity Catalog.
- **Evaluation with LLM judges** — built-in and custom scorers for banking-support quality.
- **Find & fix** — turn failing traces into eval datasets and compare prompt versions.
- **Human review** — expert labeling sessions for ground truth.
- **Production monitoring** — scheduled quality scorers.
- **Prompt optimization** — automatic prompt improvement with GEPA.
- **User feedback & Prompt Registry** — capture thumbs up/down and version prompts centrally.
