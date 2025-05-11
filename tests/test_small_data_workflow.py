"""
Integration test for the entire workflow using the small test dataset.
"""
import pytest
pytestmark = pytest.mark.integration
import os
import pandas as pd
import numpy as np
from src.logprobs import collect_logprobs
from src.analyzer import Analyzer
from src.metrics import single_evidence_estimate, pairwise_bce_of_group
from unittest.mock import patch, MagicMock


class TestSmallDataWorkflow:
    """Tests for the complete workflow with small test data."""
    
    @pytest.fixture
    def mock_llm_interface(self):
        """Create a mock LLM interface for testing."""
        mock_interface = MagicMock()

        # Create a dynamic return_value function that generates keys based on input
        def compute_logprobs_mock(prompt_data):
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

            return results

        # Set the mock to use our dynamic function
        mock_interface.compute_logprobs.side_effect = compute_logprobs_mock
        mock_interface.device = type('obj', (object,), {'type': 'cpu'})
        return mock_interface

    @patch("src.logprobs.HFInterface")
    def test_end_to_end_workflow(self, mock_hf_class, mock_llm_interface, tmp_path):
        """Test the complete workflow from data collection to BCE calculation."""
        # Set up the mock
        mock_hf_class.return_value = mock_llm_interface
        
        # Define paths
        test_data_path = "tests/data/small_test.json"
        tmp_csv_path = os.path.join(tmp_path, "test_output.csv")
        
        # 1. Collect logprobs with mocked model
        logprobs_df = collect_logprobs(
            test_data_path,
            models=["test-gpt2"],
            model_provider="hf",
            save_results=True,
            save_path=tmp_csv_path,
            verbose=False
        )
        
        # Verify logprobs collection
        assert not logprobs_df.empty, "Logprobs collection failed - empty DataFrame"
        assert "prior_logprob" in logprobs_df.columns, "Missing prior_logprob column"
        assert "likelihood_logprob" in logprobs_df.columns, "Missing likelihood_logprob column"
        assert "posterior_logprob" in logprobs_df.columns, "Missing posterior_logprob column"
        assert len(logprobs_df) == 2, f"Expected 2 rows, got {len(logprobs_df)}"
        
        # 2. Initialize analyzer
        analyzer = Analyzer(logprobs_df)
        
        # 3. Calculate evidence estimates
        analyzer = analyzer.add_column(
            "evidence_estimate",
            lambda df: single_evidence_estimate(df, "prior_logprob", "likelihood_logprob", "posterior_logprob")
        )
        
        # Verify evidence estimates
        assert "evidence_estimate" in analyzer.df.columns, "Missing evidence_estimate column"
        
        # The evidence estimates should roughly follow prior + likelihood - posterior
        for i, row in analyzer.df.iterrows():
            expected = row["prior_logprob"] + row["likelihood_logprob"] - row["posterior_logprob"]
            assert abs(row["evidence_estimate"] - expected) < 1e-6, f"Evidence estimate mismatch at row {i}"
        
        # 4. Calculate BCE
        bce_data = []
        for (class_type, evidence_text), group in analyzer.df.groupby(["class_type", "evidence_text"]):
            if len(group) < 2:
                continue
                
            bce_values = pairwise_bce_of_group(
                group, "prior_logprob", "likelihood_logprob", "posterior_logprob", square=True
            )
            
            mean_bce = np.mean(bce_values) if len(bce_values) > 0 else np.nan
            bce_data.append({
                "class_type": class_type,
                "evidence_text": evidence_text,
                "mean_bce": mean_bce
            })
        
        # Create BCE DataFrame
        bce_df = pd.DataFrame(bce_data)
        
        # Verify BCE calculation
        assert not bce_df.empty, "BCE calculation failed - empty DataFrame"
        assert "mean_bce" in bce_df.columns, "Missing mean_bce column"
        assert len(bce_df) == 1, f"Expected 1 row for BCE results, got {len(bce_df)}"
        
        # BCE should be positive (squared differences)
        assert all(bce_df["mean_bce"] > 0), "BCE values should be positive"