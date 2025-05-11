"""
Integration tests for end-to-end workflows in the Bayesian reasoning project.
"""

import pytest
import pandas as pd
import numpy as np
import os
import altair as alt
from unittest.mock import patch

from src.logprobs import collect_logprobs
from src.analyzer import Analyzer
from src.metrics import single_evidence_estimate, pairwise_bce_of_group, mean_pairwise_error_of_group
from src.visualizer import VisualisationConfig, visualize


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Tests for complete end-to-end workflows."""
    
    @pytest.fixture
    def mock_llm_interface(self):
        """Set up a mock LLM interface."""
        from unittest.mock import MagicMock

        mock_interface = MagicMock()

        # Create a dynamic return_value function that generates keys based on input
        def compute_logprobs_mock(prompt_data):
            results = {}
            for prompt_type, prompt, expected, metadata, meta_key in prompt_data:
                # Generate a realistic result tuple: (logprob, num_tokens, token_logprobs)
                if prompt_type == "prior":
                    results[meta_key] = (-5.0, 3, [-1.5, -1.8, -1.7])
                elif prompt_type == "likelihood":
                    results[meta_key] = (-3.0, 8, [-0.4, -0.3, -0.5, -0.2, -0.4, -0.6, -0.3, -0.3])
                elif prompt_type == "posterior":
                    results[meta_key] = (-4.0, 3, [-1.2, -1.3, -1.5])
            return results

        # Set the mock to use our dynamic function
        mock_interface.compute_logprobs.side_effect = compute_logprobs_mock
        mock_interface.device = type('obj', (object,), {'type': 'cpu'})
        return mock_interface
    
    @patch("src.logprobs.HFInterface")
    def test_full_pipeline(self, mock_hf_class, mock_llm_interface, test_data_json, tmp_path):
        """Test the full pipeline from data collection to visualization."""
        # Set up the mock
        mock_hf_class.return_value = mock_llm_interface
        
        # 1. Collect logprobs
        tmp_csv_path = os.path.join(tmp_path, "test_output.csv")
        logprobs_df = collect_logprobs(
            test_data_json,
            models=["test-model"],
            model_provider="hf",
            save_results=True,
            save_path=tmp_csv_path,
            verbose=False
        )
        
        # Check that we got results
        assert not logprobs_df.empty
        assert os.path.exists(tmp_csv_path)
        
        # 2. Load the data with Analyzer
        analyzer = Analyzer(tmp_csv_path)
        
        # 3. Calculate evidence estimates
        analyzer = analyzer.add_column(
            "evidence_estimate",
            lambda df: single_evidence_estimate(df, "prior_logprob", "likelihood_logprob", "posterior_logprob")
        )
        
        # Check that evidence estimates were calculated
        assert "evidence_estimate" in analyzer.df.columns
        
        # 4. Calculate BCE for each group
        bce_data = []
        for (class_type, evidence_text), group in analyzer.df.groupby(["class_type", "evidence_text"]):
            # Skip groups with less than 2 items
            if len(group) < 2:
                continue
                
            # Get BCE values
            bce_values = pairwise_bce_of_group(
                group, "prior_logprob", "likelihood_logprob", "posterior_logprob"
            )
            
            # Calculate mean BCE
            mean_bce = np.mean(bce_values) if bce_values else np.nan
            
            # Store results
            bce_data.append({
                "class_type": class_type,
                "evidence_text": evidence_text,
                "mean_bce": mean_bce
            })
        
        # Create DataFrame with BCE results
        bce_df = pd.DataFrame(bce_data)
        
        # Check that we have BCE results
        assert not bce_df.empty
        
        # 5. Create a visualization config
        config = VisualisationConfig(
            plot_fn=alt.Chart.mark_bar,
            x_category="class_type",
            y_category="mean_bce",
            fig_title="Bayesian Consistency Error by Class Type"
        )
        
        # 6. Generate a visualization
        chart = visualize(bce_df, config)
        
        # Check that a chart was created
        assert chart is not None
        assert isinstance(chart, alt.Chart)
        assert chart.mark == "bar"
        
        # 7. Save the chart to a file
        chart_path = os.path.join(tmp_path, "bce_chart.html")
        chart.save(chart_path)
        
        # Check that the chart was saved
        assert os.path.exists(chart_path)
    
    @patch("src.logprobs.HFInterface")
    def test_cross_model_comparison(self, mock_hf_class, mock_llm_interface, test_data_json, tmp_path):
        """Test a workflow that compares BCE across different models."""
        # Set up the mock with different responses for each model
        def mock_compute_logprobs(prompt_data):
            results = {}
            for prompt_type, prompt, expected, metadata, meta_key in prompt_data:
                # Return different values based on the current model name
                if mock_hf_class.return_value.model_name == "model-1":
                    if prompt_type == "prior":
                        results[meta_key] = (-5.0, 3, [-1.5, -1.8, -1.7])
                    elif prompt_type == "likelihood":
                        results[meta_key] = (-3.0, 8, [-0.4, -0.3, -0.5, -0.2, -0.4, -0.6, -0.3, -0.3])
                    elif prompt_type == "posterior":
                        results[meta_key] = (-4.0, 3, [-1.2, -1.3, -1.5])
                else:
                    if prompt_type == "prior":
                        results[meta_key] = (-4.5, 3, [-1.3, -1.6, -1.6])
                    elif prompt_type == "likelihood":
                        results[meta_key] = (-2.5, 8, [-0.3, -0.3, -0.4, -0.2, -0.3, -0.5, -0.2, -0.3])
                    elif prompt_type == "posterior":
                        results[meta_key] = (-3.5, 3, [-1.1, -1.2, -1.2])
            return results
        
        mock_llm_interface.compute_logprobs.side_effect = mock_compute_logprobs
        mock_llm_interface.device = type('obj', (object,), {'type': 'cpu'})
        mock_hf_class.return_value = mock_llm_interface
        
        # 1. Collect logprobs for multiple models
        tmp_csv_path = os.path.join(tmp_path, "multi_model_output.csv")
        logprobs_df = collect_logprobs(
            test_data_json,
            models=["model-1", "model-2"],
            model_params=[{"temperature": 1.0}, {"temperature": 1.0}],
            param_mapping_strategy="one_to_one",
            model_provider="hf",
            save_results=True,
            save_path=tmp_csv_path,
            verbose=False
        )
        
        # Check that we got results for both models
        assert not logprobs_df.empty
        assert "model_name" in logprobs_df.columns
        assert set(logprobs_df["model_name"].unique()) == {"model-1", "model-2"}
        
        # 2. Load the data with Analyzer
        analyzer = Analyzer(tmp_csv_path)
        
        # 3. Calculate evidence estimates
        analyzer = analyzer.add_column(
            "evidence_estimate",
            lambda df: single_evidence_estimate(df, "prior_logprob", "likelihood_logprob", "posterior_logprob")
        )
        
        # 4. Calculate BCE grouped by model and class_type
        bce_data = []
        for (model_name, class_type, evidence_text), group in analyzer.df.groupby(["model_name", "class_type", "evidence_text"]):
            # Skip groups with less than 2 items
            if len(group) < 2:
                continue
                
            # Get all pairwise BCE values
            bce_values = pairwise_bce_of_group(
                group, "prior_logprob", "likelihood_logprob", "posterior_logprob"
            )
            
            # Calculate mean BCE
            mean_bce = np.mean(bce_values) if bce_values else np.nan
            
            # Store results
            bce_data.append({
                "model_name": model_name,
                "class_type": class_type,
                "evidence_text": evidence_text,
                "mean_bce": mean_bce
            })
        
        # Create DataFrame with BCE results
        bce_df = pd.DataFrame(bce_data)
        
        # Check that we have BCE results
        assert not bce_df.empty
        assert "model_name" in bce_df.columns
        
        # 5. Create a visualization config for comparison
        config = VisualisationConfig(
            plot_fn=alt.Chart.mark_bar,
            x_category="model_name",
            y_category="mean_bce",
            color_category="class_type",
            fig_title="Bayesian Consistency Error by Model and Class Type"
        )
        
        # 6. Generate a visualization
        chart = visualize(bce_df, config)
        
        # Check that a chart was created
        assert chart is not None
        assert isinstance(chart, alt.Chart)
        
        # 7. Save the chart to a file
        chart_path = os.path.join(tmp_path, "model_comparison_chart.html")
        chart.save(chart_path)
        
        # Check that the chart was saved
        assert os.path.exists(chart_path)