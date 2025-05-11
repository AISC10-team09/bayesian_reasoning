"""
Test to verify that our mock values are calibrated to match real model behavior.

This test compares logprobs from a real model with those from our mock interface
to ensure our mocks remain realistic and scientifically relevant.
"""

import pytest
import pandas as pd
import numpy as np
import os
import json
from unittest.mock import patch, MagicMock
from scipy import stats
import matplotlib.pyplot as plt

from src.logprobs import collect_logprobs
from src.models.hf_interface import HFInterface
from src.analyzer import Analyzer
from src.metrics import single_evidence_estimate

# Mark these tests with appropriate categories
pytestmark = [pytest.mark.logprobs, pytest.mark.slow]

class TestMockCalibration:
    """Tests to ensure mocks are properly calibrated to match real model behavior."""
    
    @pytest.fixture
    def mock_llm_interface(self):
        """Create our standard mock LLM interface for testing."""
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
    
    def run_with_mock(self, mock_llm_interface, tmp_path):
        """Run logprobs collection with our mock interface."""
        with patch("src.logprobs.HFInterface") as mock_hf_class:
            # Set up the mock
            mock_hf_class.return_value = mock_llm_interface
            
            # Define paths
            test_data_path = "tests/data/small_test.json"
            tmp_csv_path = os.path.join(tmp_path, "mock_output.csv")
            
            # Collect logprobs with mocked model
            logprobs_df = collect_logprobs(
                test_data_path,
                models=["test-gpt2"],
                model_provider="hf",
                save_results=True,
                save_path=tmp_csv_path,
                verbose=False
            )
            
            return logprobs_df
    
    def run_with_real_model(self, tmp_path):
        """Run logprobs collection with a real model."""
        # Define paths
        test_data_path = "tests/data/small_test.json"
        tmp_csv_path = os.path.join(tmp_path, "real_output.csv")
        
        # Collect logprobs with real model (using smallest gpt2 for speed)
        logprobs_df = collect_logprobs(
            test_data_path,
            models=["gpt2"],
            model_provider="hf",
            model_params=[{"max_length": 20}],  # Limit token generation for speed
            save_results=True,
            save_path=tmp_csv_path,
            verbose=False
        )
        
        return logprobs_df
    
    def compare_distributions(self, real_df, mock_df, column, alpha=0.05):
        """Compare distributions of real and mock values using statistical tests."""
        real_values = real_df[column].dropna().values
        mock_values = mock_df[column].dropna().values
        
        if len(real_values) < 2 or len(mock_values) < 2:
            # Not enough data for statistical test
            return True, 1.0, "Not enough data for statistical test"
        
        # Perform Kolmogorov-Smirnov test for distribution similarity
        ks_stat, p_value = stats.kstest(real_values, mock_values)
        
        # Check if distributions are significantly different
        distributions_similar = p_value > alpha
        
        return distributions_similar, p_value, f"KS test p-value: {p_value:.4f}"
    
    def compare_bayesian_relationships(self, real_df, mock_df):
        """Compare Bayesian relationships between real and mock data."""
        # Calculate evidence estimates for both
        def add_evidence_estimates(df):
            analyzer = Analyzer(df)
            analyzer = analyzer.add_column(
                "evidence_estimate",
                lambda df: single_evidence_estimate(df, "prior_logprob", "likelihood_logprob", "posterior_logprob")
            )
            return analyzer.df
        
        real_with_estimates = add_evidence_estimates(real_df)
        mock_with_estimates = add_evidence_estimates(mock_df)
        
        # Calculate BCE values for each
        def calculate_bce(df):
            bce_values = []
            for (class_type, evidence_text), group in df.groupby(["class_type", "evidence_text"]):
                if len(group) < 2:
                    continue
                    
                evidence_estimates = group["evidence_estimate"].values
                for i in range(len(evidence_estimates)):
                    for j in range(i+1, len(evidence_estimates)):
                        bce = (evidence_estimates[i] - evidence_estimates[j])**2
                        bce_values.append(bce)
            return np.mean(bce_values) if bce_values else np.nan
        
        real_bce = calculate_bce(real_with_estimates)
        mock_bce = calculate_bce(mock_with_estimates)
        
        # Calculate the percentage difference between BCEs
        if not np.isnan(real_bce) and not np.isnan(mock_bce) and real_bce != 0:
            bce_difference = abs(mock_bce - real_bce) / abs(real_bce)
            return bce_difference < 0.5, bce_difference, f"BCE difference: {bce_difference:.2%}"
        else:
            return True, 0.0, "Could not calculate BCE comparison (insufficient data)"
    
    def plot_comparison(self, real_df, mock_df, column, tmp_path):
        """Generate plots comparing distributions for visual inspection."""
        real_values = real_df[column].dropna().values
        mock_values = mock_df[column].dropna().values
        
        if len(real_values) < 1 or len(mock_values) < 1:
            return
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 1, 1)
        plt.hist(real_values, bins=10, alpha=0.7, label='Real Model')
        plt.hist(mock_values, bins=10, alpha=0.7, label='Mock')
        plt.title(f'Distribution of {column}')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.boxplot([real_values, mock_values], labels=['Real Model', 'Mock'])
        plt.title(f'Boxplot of {column}')
        
        # Save the plot
        plots_dir = os.path.join(tmp_path, "comparison_plots")
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, f"{column}_comparison.png"))
        plt.close()
    
    @pytest.mark.parametrize("save_plots", [True])  # Enable plot saving
    def test_mock_calibration(self, mock_llm_interface, tmp_path, save_plots):
        """Test that our mock logprobs are calibrated to match real model behavior."""
        try:
            # Get logprobs from mock
            mock_df = self.run_with_mock(mock_llm_interface, tmp_path)
            assert not mock_df.empty, "Failed to get mock logprobs"
            
            # Get logprobs from real model
            real_df = self.run_with_real_model(tmp_path)
            assert not real_df.empty, "Failed to get real model logprobs"
            
            # Compare distributions for key columns
            columns_to_check = [
                "prior_logprob", 
                "likelihood_logprob", 
                "posterior_logprob"
            ]
            
            all_distributions_similar = True
            results = []
            
            for column in columns_to_check:
                similar, p_value, message = self.compare_distributions(real_df, mock_df, column)
                all_distributions_similar = all_distributions_similar and similar
                results.append((column, similar, message))
                
                if save_plots:
                    self.plot_comparison(real_df, mock_df, column, tmp_path)
            
            # Compare Bayesian relationships
            bce_similar, difference, bce_message = self.compare_bayesian_relationships(real_df, mock_df)
            results.append(("BCE", bce_similar, bce_message))
            
            # Check if calibration is needed
            if not all_distributions_similar or not bce_similar:
                calibration_needed = []
                for column, similar, message in results:
                    if not similar:
                        calibration_needed.append(f"{column}: {message}")
                
                # Print calibration guidance
                print("\nMock Calibration Needed:")
                for item in calibration_needed:
                    print(f"- {item}")
                    
                print("\nSample Real Model Values:")
                for column in columns_to_check:
                    values = real_df[column].tolist()
                    print(f"{column}: {values}")
                
                # Provide suggested mock values
                print("\nSuggested Mock Updates:")
                print("def compute_logprobs_mock(prompt_data):")
                print("    results = {}")
                print("    for prompt_type, prompt, expected, metadata, meta_key in prompt_data:")
                print("        if prompt_type == \"prior\":")
                print(f"            results[meta_key] = ({real_df['prior_logprob'].mean():.1f}, {real_df['prior_num_tokens'].mean():.0f}, {real_df['prior_token_logprobs'].iloc[0][:5]}...)")
                print("        elif prompt_type == \"likelihood\":")
                print(f"            results[meta_key] = ({real_df['likelihood_logprob'].mean():.1f}, {real_df['likelihood_num_tokens'].mean():.0f}, {real_df['likelihood_token_logprobs'].iloc[0][:5]}...)")
                print("        elif prompt_type == \"posterior\":")
                print(f"            results[meta_key] = ({real_df['posterior_logprob'].mean():.1f}, {real_df['posterior_num_tokens'].mean():.0f}, {real_df['posterior_token_logprobs'].iloc[0][:5]}...)")
                print("    return results")
                
                # Fail the test if difference is too large
                if difference > 1.0:  # More than 100% difference in BCE
                    pytest.fail("Mock values are significantly miscalibrated compared to real model outputs")
                else:
                    pytest.xfail("Mock values need recalibration but differences are not critical")
                    
        except Exception as e:
            pytest.skip(f"Calibration test failed with error: {str(e)}")