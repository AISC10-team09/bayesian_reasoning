"""
conftest.py - Shared fixtures for pytest tests in the Bayesian reasoning project.

This module contains fixtures that can be reused across different test files,
including test data loaders, mock model implementations, and utility functions.
"""

import os
import json
import pandas as pd
import pytest
import numpy as np

# Define test data paths relative to this file
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")
TEST_DATA_JSON = os.path.join(TEST_DATA_DIR, "test_data.json")
TEST_LOGPROBS_CSV = os.path.join(TEST_DATA_DIR, "test_logprobs.csv")

# Create the test data directory if it doesn't exist
os.makedirs(TEST_DATA_DIR, exist_ok=True)


# --- Data Fixtures ---

@pytest.fixture(scope="session")
def test_data_json():
    """Load the test data JSON file."""
    with open(TEST_DATA_JSON, "r") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def test_logprobs_df():
    """Load the test logprobs CSV as a DataFrame."""
    return pd.read_csv(TEST_LOGPROBS_CSV)


@pytest.fixture(scope="session")
def sample_bayesian_item():
    """Return a single Bayesian reasoning item for simple tests."""
    return {
        "class_type": "writing_style",
        "conversation_history": "We've been discussing different writing styles.",
        "candidate_classes": [" style A.", " style B."],
        "class_elicitation": " My style is",
        "evidence_elicitation": " I write with",
        "evidence": [{"category": "test", "evidence_text": " particular characteristics."}]
    }


# --- Mock Model Interface ---

class MockLLMInterface:
    """Mock implementation of LLMInterface for testing.

    This mock uses token-level log probabilities calibrated from real GPT-2 model runs.
    It produces realistic token counts and log probability distributions.
    """

    def __init__(self, model_name="test-model", **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.device = type('obj', (object,), {'type': 'cpu'})  # Mock device object

    def compute_logprobs(self, prompt_data):
        """Return realistic logprobs based on real model behavior.

        Arguments:
            prompt_data: List of (type, prompt, expected, meta, meta_key) tuples

        Returns:
            Dictionary mapping meta_keys to (logprob, num_tokens, token_logprobs) tuples
        """
        results = {}

        for prompt_type, prompt, expected, metadata, meta_key in prompt_data:
            # Generate a realistic result tuple based on prompt type and expected text
            if prompt_type == "prior":
                if "aphasia" in expected:
                    results[meta_key] = (-40.1, 7, [-8.2, -9.5, -0.3, -5.3, -8.2, -3.6, -5.1])
                elif "slang" in expected:
                    results[meta_key] = (-31.5, 5, [-14.8, -3.8, -4.0, -4.4, -4.5])
                else:
                    # Fallback for unknown classes with similar statistical properties
                    results[meta_key] = (-35.0, 6, [-6.0, -7.0, -6.0, -5.0, -6.0, -5.0])

            elif prompt_type == "likelihood":
                if "finna" in expected:
                    results[meta_key] = (-55.6, 11, [-16.4, -2.0, -4.3, -4.2, -5.4, -8.8, -6.7, -2.4, -4.3, -0.6, -0.7])
                else:
                    # Fallback for other evidence with similar token count
                    results[meta_key] = (-50.0, 10, [-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0])

            elif prompt_type == "posterior":
                if "aphasia" in expected:
                    results[meta_key] = (-40.0, 7, [-7.7, -10.0, -0.2, -5.4, -8.3, -3.9, -4.6])
                elif "slang" in expected:
                    results[meta_key] = (-31.7, 5, [-14.3, -4.2, -4.6, -4.4, -4.3])
                else:
                    # Fallback for unknown classes
                    results[meta_key] = (-35.0, 6, [-5.8, -6.8, -5.8, -4.8, -5.8, -6.0])

            else:
                # Fallback for unknown prompt types
                length = len(expected) // 4  # Approximate token count
                if length < 3: length = 3
                total_logprob = -5.0 * length  # Reasonable average
                token_logprobs = [-5.0] * length  # Uniform distribution
                results[meta_key] = (total_logprob, length, token_logprobs)

        return results
    
    def release(self):
        """Mock resource release."""
        pass


@pytest.fixture
def mock_llm_interface():
    """Return a mock LLM interface for testing."""
    return MockLLMInterface()


# --- Utility Fixtures ---

@pytest.fixture
def set_random_seed():
    """Set random seeds for reproducible tests."""
    np.random.seed(42)
    
    
@pytest.fixture
def tmp_csv_path(tmp_path):
    """Provide a temporary CSV path for file output tests."""
    return tmp_path / "test_output.csv"