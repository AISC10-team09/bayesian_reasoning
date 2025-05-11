"""
Smoke test with a real model to verify the entire pipeline works with an actual model.
This test is marked as 'slow' since it loads a real model.
"""

import pytest
import pandas as pd
import os
from src.logprobs import collect_logprobs
from src.analyzer import Analyzer
from src.metrics import single_evidence_estimate

# Mark this test as slow since it loads a real model
pytestmark = pytest.mark.slow

@pytest.mark.skipif(not os.path.exists("tests/data/small_test.json"),
                   reason="small_test.json not found, skipping real model test")
def test_real_model_smoke_gpt2_tiny():
    """Smoke test with the smallest available GPT2 model."""
    # This test will be skipped by default due to the 'slow' mark
    # Run with: python -m pytest tests/test_real_model_smoke.py -v
    
    # Define paths
    test_data_path = "tests/data/small_test.json"
    result_path = "data/test_real_tiny_logprobs.csv"
    
    try:
        # Try to collect logprobs with a tiny tokenizer limit to make it fast
        logprobs_df = collect_logprobs(
            test_data_path,
            models=["gpt2"],
            model_provider="hf",
            model_params=[{"max_length": 20}],  # Limit token generation for speed
            save_results=True,
            save_path=result_path,
            verbose=False
        )
        
        assert not logprobs_df.empty, "Logprobs collection failed - empty DataFrame"
        
        # Minimal analysis to verify the pipeline
        analyzer = Analyzer(logprobs_df)
        
        # Calculate evidence estimates
        analyzer = analyzer.add_column(
            "evidence_estimate",
            lambda df: single_evidence_estimate(df, "prior_logprob", "likelihood_logprob", "posterior_logprob")
        )
        
        assert "evidence_estimate" in analyzer.df.columns, "Evidence estimate calculation failed"
        
        # Test passes if we reach this point without errors
        
    except Exception as e:
        # If an error occurs, we want to know about it but still clean up
        pytest.fail(f"Real model test failed with error: {str(e)}")
        
    finally:
        # Clean up the created file
        if os.path.exists(result_path):
            try:
                os.remove(result_path)
            except:
                pass  # If cleanup fails, don't worry about it