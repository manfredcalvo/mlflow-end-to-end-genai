"""Agent module for the banking workshop.

This module contains the shared banking customer-support agent (used by BOTH
the workshop notebooks and the app's FastAPI server).

Import directly from submodules to avoid triggering heavy initialization:
  from mlflow_demo.agent.agent import AGENT, BankingAgent, get_agent
  from mlflow_demo.agent.prompts import ORIGINAL_PROMPT_TEMPLATE
"""
