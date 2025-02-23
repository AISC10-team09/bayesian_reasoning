"""
config.py

This module holds global configuration settings for the Bayesian consistency experiment project.
It includes parameters for LLM access (API key, model name, backend), generation settings,
directory paths for saving experiment outputs and logs, and default values for experiments.

Configuration values can be overridden by setting corresponding environment variables.
"""

import os

# --- LLM Configuration ---
# API key for OpenAI (if using the "openai" backend). Set your API key as an environment variable or update the default.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "our-api-key-will-be-here")

# Model name to use. For local testing, a lightweight model like "gpt2" is used by default.
MODEL_NAME = os.getenv("MODEL_NAME", "gpt2")

# Backend to use: "openai" for API calls, "local" for using local Hugging Face models.
LLM_BACKEND = os.getenv("LLM_BACKEND", "local")

# Generation parameters for the LLM.
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "50"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
LOGPROBS = int(os.getenv("LOGPROBS", "5"))

# --- Experiment and Output Configuration ---
# Directory where experimental results will be saved.
EXPERIMENT_RESULTS_DIR = os.getenv("EXPERIMENT_RESULTS_DIR", "results/")
# Directory for log files.
LOGS_DIR = os.getenv("LOGS_DIR", "logs/")

# --- Default Experiment Values ---
# Default conversation history for experiments.
DEFAULT_HISTORY = "We've been discussing various topics and ideas."
# Default candidate classes for testing Bayesian consistency.
DEFAULT_CANDIDATE1 = "Shakespeare"
DEFAULT_CANDIDATE2 = "Mark Twain"
# Default evidence snippet.
DEFAULT_EVIDENCE = "To thine own self be true."
