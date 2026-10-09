# BGP MLflow 3 GenAI Workshop — Facilitation Guide

## Overview

A hands-on workshop for 20 participants from BGP Bank (Panama) covering the full MLflow 3 GenAI quality lifecycle: tracing → evaluation → labeling → prompt optimization → monitoring. Each participant deploys their own banking customer support agent as a Databricks App, works through 6 notebooks with exercises, optimizes the agent's prompt, and redeploys.

**Everything runs in Databricks.** Participants need only the Databricks CLI + a repo clone — no local Python environments, no `.env` files, no npm/uv builds. All setup runs as serverless jobs in the workspace, watched cell-by-cell in the Jobs UI.

## Prerequisites

### Workspace
- Databricks workspace with Unity Catalog enabled
- Databricks Apps enabled (`databricks apps list` should work)
- Model Serving + AI Gateway enabled (Claude Sonnet 5 pay-per-token available)
- SQL warehouse (Small or Medium) for MLflow tracing — its ID is the `warehouse_id` default in `databricks.yml`
- 20 participant users with workspace access

### Participant machines
- **Databricks CLI v1.19.0+** configured (`databricks auth login`) — v1.19.0+ is required because the agent's LLM is served through an AI Gateway `model_services` DAB resource (Beta) that older CLIs silently ignore. Verify with `databricks --version`.
- Git (to clone the repo)
- That's all — no Python, uv, or Node needed.

## Pre-Workshop Setup

Participants self-provision their own data at a **single shared schema**, namespaced by a username prefix — nobody needs rights to create catalogs or schemas. There is no `p01..p20` schema and no central data load.

### Admin (one-time)
A workspace admin provisions one catalog + one schema and grants access:
```sql
-- 1. Create the shared location (admin only)
CREATE CATALOG IF NOT EXISTS workshop_bgp;
CREATE SCHEMA  IF NOT EXISTS workshop_bgp.shared_data;

-- 2. Let every participant create THEIR OWN prefixed objects in it
GRANT USE CATALOG ON CATALOG workshop_bgp TO `<each participant>`;
GRANT USE SCHEMA, CREATE TABLE, CREATE FUNCTION ON SCHEMA workshop_bgp.shared_data TO `<each participant>`;
```
> **That's all the admin does.** The app's service principal needs no manual grants:
> the app *declares* its UC access as DAB app resources (warehouse, experiment, and the
> MLflow trace tables — all created before the app during `bundle deploy`), and the
> `bgp_participant_data` job — running as the participant, who owns their
> `<prefix>_get_*` functions — grants the app SP `EXECUTE` on them right after creating them.

### Instructor
1. Hand out the **workspace URL** and confirm the `warehouse_id` default in `databricks.yml` (and the `warehouse` widget default in `mlflow_demo/utils/mlflow_helpers.py`) point at the workshop's shared SQL warehouse.
2. (Optional) Run `workshop/instructor_setup.py` on a cluster to **preflight** — it verifies the schema exists and is writable and prints the admin grant checklist. It does **not** create anything.

## Workshop Timeline (~4 hours)

| Time | Phase | Activity | Duration |
|------|-------|----------|----------|
| 0:00 | Intro | Welcome, overview of MLflow GenAI | 15 min |
| 0:15 | Deploy | Each participant runs the 4 self-service CLI commands | 30 min |
| 0:45 | Watch | Data job + apps finishing in the Jobs UI | 15 min |
| 1:00 | Use | Participants interact with their app, submit feedback | 30 min |
| 1:30 | Notebook 1 | Observe with Tracing | 30 min |
| 2:00 | Notebook 2 | Create Quality Metrics | 30 min |
| 2:30 | Break | | 15 min |
| 2:45 | Notebook 3 | Find & Fix Quality Issues | 30 min |
| 3:15 | Notebook 4 | Human Review | 20 min |
| 3:35 | Notebook 5 | Production Monitoring | 15 min |
| 3:50 | Optimize | Notebook 6: GEPA optimization + redeploy | 25 min |
| 4:15 | Wrap | Q&A, what to take home | 15 min |

## Phase Details

### Phase 1: Deploy (self-service, 4 commands)

Each participant, from their clone (everything else is auto-derived from their username):
```bash
databricks auth login --host https://<workspace>.cloud.databricks.com --profile bgp

# 1. Create their app config, MLflow experiment, AI Gateway model service,
#    and the self-service data job. No local builds — the built frontend
#    is committed in app/client/build/.
#    NOTE: this first deploy reports an EXPECTED error on the Genie space
#    (Genie validates that its tables exist, and the bank tables are only
#    created by the data job below). The other resources deploy fine.
databricks bundle deploy --target dev --profile bgp

# 2. Create their <prefix>_bank_* tables + <prefix>_get_* UC functions
#    (watch each notebook cell execute in the Jobs UI)
databricks bundle run bgp_participant_data --target dev --profile bgp

# 3. Deploy again — now the Genie space (example Genie agent over the
#    participant's banking data) creates cleanly
databricks bundle deploy --target dev --profile bgp

# 4. Deploy + start their app
databricks bundle run bgp_agent --target dev --profile bgp
```

The prompt, sample traces, and baseline evaluations are **part of the workshop itself**:
importing the agent (notebook 1, step 1) registers the `<prefix>_bgp_support_prompt` if
it doesn't exist (the app falls back to its built-in prompt until then), notebook 1's
step 4 generates the 10 sample traces, and notebook 2's evaluate run attaches the judges'
assessments that the later notebooks build on.

What gets derived from their username (e.g. `jane.doe@x.com` → prefix `jane_doe`):
their `<prefix>_bank_*` tables, `<prefix>_get_*` functions, `<prefix>_bgp_support_prompt`,
`<prefix>_bgp_agent_llm` model service, `bgp-agent-<prefix>` app, and the
`/Users/<you>/bgp_bank_workshop` MLflow experiment. Zero `--var` flags needed; the only
common override is `--var app_name=bgp-agent-<shorter-name>` when a username is long
or contains underscores (app names: lowercase/hyphens/≤30 chars).

**Troubleshooting:**
- "App won't start" → check app logs (Compute → Apps → `<app>` → logs)
- Data job fails on `CREATE TABLE/FUNCTION` → the admin hasn't granted you create rights on the schema
- Agent can't call functions → re-run `bundle run bgp_participant_data` (it (re)grants the app SP `EXECUTE` on your functions)
- No traces in MLflow → the app declares its trace-table access as DAB resources; check the app's Compute → resources list, and that `databricks --version` ≥ 1.19
- Long/underscored username → pass `--var app_name=...`

### Phase 2: Use & Feedback (30 min)

Participants visit their app URL, ask banking questions, and submit thumbs up/down feedback. They then open the MLflow experiment to see traces with tool call spans.

### Phase 3: Notebook Labs (90 min)

Participants open notebooks 0–6 **in the workspace** at
`/Workspace/Users/<you>/bgp_bank_workshop/notebooks/` (synced by their deploy). Each
notebook auto-configures itself from their username — the widgets only carry the shared
catalog/schema/warehouse. **Run them in order**: notebook 1 generates the sample traces
(and registers the prompt), notebook 2 attaches the judges' assessments — everything the
later notebooks depend on.

### Phase 4: Optimize & Redeploy (30 min)

Participants run notebook 6 (GEPA optimization) and register the improved prompt with the
`production` alias. To make the live app pick it up, one command:
```bash
databricks bundle run bgp_agent --target dev --profile bgp
```
Then ask the same question in the app and compare.

## CI/CD Demo (5 min)

At the end of the workshop, the instructor demonstrates the CI/CD story:

1. **DAB definition** (`databricks.yml` + `resources/*.yml`): the app, model service, experiment, and jobs are all declarative — no shell scripts, no sed, no manual substitution.
2. **Workshop flow** (what participants just did): `bundle deploy` + `bundle run <job>` from a clone, everything derived from the username.
3. **Production flow**: push to `main` triggers `.github/workflows/deploy.yml`, which builds the frontend and runs the same `bundle deploy/run --target prod` with secrets. An equivalent **CircleCI** example lives in `.circleci/config.yml` — identical steps and variables, showing the bundle is CI-provider-agnostic (a good customer talking point).
4. Key message: "Once your DAB is set up, CI/CD is just `databricks bundle deploy` — no shell scripts, no sed, no manual steps"

## Troubleshooting Guide

| Issue | Solution |
|-------|----------|
| App won't start | Check the app logs (Compute → Apps → your app), verify `databricks --version` ≥ 1.19.0 |
| No traces in MLflow | The app's trace-table access is a DAB app resource — check Compute → Apps → your app → resources |
| UC function error | Re-run `bundle run bgp_participant_data` — it (re)grants the app SP `EXECUTE` on your functions |
| Scorer collision | Scorers are namespaced with your username prefix automatically |
| LLM rate limits | If 20 participants hit rate limits, stagger the jobs or use provisioned throughput |
| DAB validation fails | Run `databricks bundle validate --target dev` |
| Notebooks can't find the experiment | Run `databricks bundle deploy` first — it creates your MLflow experiment (notebooks find it by searching, robust to dev-mode name prefixes) |

## Workshop Materials

| Material | Location |
|----------|----------|
| Instructor preflight | `workshop/instructor_setup.py` |
| Banking data generator (job notebook) | `workshop/generate_banking_data.py` |
| UC functions (job notebook) | `workshop/create_uc_functions.py` |
| Sample questions (used in Notebook 1) | `workshop/banking_questions.jsonl` |
| Notebooks 0-6 (repo) | `notebooks/` |
| Notebooks 0-6 (in workspace after deploy) | `/Workspace/Users/<you>/bgp_bank_workshop/notebooks/` |
| Example Genie agent (space resource) | `resources/bgp_genie_space.yml` |
| Embedded AI/BI dashboard (resource + Analytics tab) | `resources/bgp_dashboard.yml`, `app/client/src/components/analytics/AnalyticsTab.tsx` |
| Agent code | `mlflow_demo/agent/agent.py` |
| Banking prompts | `mlflow_demo/agent/prompts.py` |
| Banking scorers | `mlflow_demo/evaluation/evaluator.py` |
| Notebook env helper | `mlflow_demo/utils/mlflow_helpers.py` |
