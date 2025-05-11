"""
Unit tests for the metrics module in src/metrics.py
"""

import pytest
import pandas as pd
import numpy as np
from src.metrics import (
    single_evidence_estimate,
    pairwise_error_of_group,
    mean_pairwise_error_of_group,
    pairwise_bce_of_group
)


class TestSingleEvidenceEstimate:
    """Tests for the single_evidence_estimate function."""
    
    def test_basic_calculation(self):
        """Test basic evidence estimate calculation."""
        # Create a sample DataFrame with known values
        df = pd.DataFrame({
            "log_prior": [-5.0, -6.0],
            "log_likelihood": [-3.0, -2.0],
            "log_posterior": [-4.0, -3.0]
        })
        
        # Calculate evidence estimate
        result = single_evidence_estimate(df, "log_prior", "log_likelihood", "log_posterior")
        
        # Check results: log_prior + log_likelihood - log_posterior
        assert len(result) == 2
        assert result[0] == -5.0 + (-3.0) - (-4.0) == -4.0
        assert result[1] == -6.0 + (-2.0) - (-3.0) == -5.0
        
    def test_with_test_logprobs(self, test_logprobs_df):
        """Test evidence estimate with the test logprobs data."""
        # Group by class_type and apply function
        for (class_type, evidence_text), group in test_logprobs_df.groupby(["class_type", "evidence_text"]):
            result = single_evidence_estimate(
                group, "prior_logprob", "likelihood_logprob", "posterior_logprob"
            )
            
            # Check results match expected values from README
            if class_type == "writing_style":
                # Check first row (aphasia-affected writing)
                assert abs(result.iloc[0] - (-4.0)) < 1e-6
                # Check second row (slang-based writing)
                assert abs(result.iloc[1] - (-5.0)) < 1e-6
            elif class_type == "architectural_styles":
                # Check first row (gothic)
                assert abs(result.iloc[0] - (-3.0)) < 1e-6
                # Check second row (beaux arts)
                assert abs(result.iloc[1] - (-4.5)) < 1e-6


class TestPairwiseErrorOfGroup:
    """Tests for the pairwise_error_of_group function."""
    
    def test_squared_errors(self):
        """Test calculation of squared pairwise errors."""
        # Create a sample DataFrame with a column of values
        df = pd.DataFrame({"value": [1.0, 2.0, 4.0]})
        
        # Calculate pairwise squared errors
        result = pairwise_error_of_group(df, "value", square=True)
        
        # Check results - should have (3 choose 2) = 3 pairs
        # (1-2)^2 = 1, (1-4)^2 = 9, (2-4)^2 = 4
        assert len(result) == 3
        assert sorted(result) == [1.0, 4.0, 9.0]
        
    def test_absolute_errors(self):
        """Test calculation of absolute pairwise errors."""
        # Create a sample DataFrame with a column of values
        df = pd.DataFrame({"value": [1.0, 2.0, 4.0]})
        
        # Calculate pairwise absolute errors
        result = pairwise_error_of_group(df, "value", square=False)
        
        # Check results - should have (3 choose 2) = 3 pairs
        # |1-2| = 1, |1-4| = 3, |2-4| = 2
        assert len(result) == 3
        assert sorted(result) == [1.0, 2.0, 3.0]
        
    def test_with_nan_values(self):
        """Test handling of NaN values."""
        # Create a sample DataFrame with some NaN values
        df = pd.DataFrame({"value": [1.0, np.nan, 4.0]})
        
        # Calculate pairwise errors, NaNs should be dropped
        result = pairwise_error_of_group(df, "value")
        
        # Check results - should have only one valid pair
        # (1-4)^2 = 9
        assert len(result) == 1
        assert result[0] == 9.0
        
    def test_with_single_value(self):
        """Test with only one valid value (should return np.nan as there are no pairs)."""
        # Create a sample DataFrame with only one value
        df = pd.DataFrame({"value": [1.0]})

        # Calculate pairwise errors, should return np.nan as there are no pairs
        result = pairwise_error_of_group(df, "value")

        # Check results - the function should return np.nan when there's only one value
        # Because MSE requires at least 2 data points
        assert np.isnan(result), "Expected np.nan for single value input"
        
    def test_with_evidence_estimates(self, test_logprobs_df):
        """Test using the evidence estimates from the test logprobs."""
        # First calculate evidence estimates
        test_logprobs_df["evidence_estimate"] = single_evidence_estimate(
            test_logprobs_df, "prior_logprob", "likelihood_logprob", "posterior_logprob"
        )
        
        # Calculate pairwise errors for writing_style
        writing_style_group = test_logprobs_df[test_logprobs_df["class_type"] == "writing_style"]
        writing_errors = pairwise_error_of_group(writing_style_group, "evidence_estimate")
        
        # Calculate pairwise errors for architectural_styles
        arch_style_group = test_logprobs_df[test_logprobs_df["class_type"] == "architectural_styles"]
        arch_errors = pairwise_error_of_group(arch_style_group, "evidence_estimate")
        
        # Check results
        # For writing_style: (-4.0) - (-5.0) = 1.0, squared = 1.0
        assert len(writing_errors) == 1
        assert abs(writing_errors[0] - 1.0) < 1e-6
        
        # For architectural_styles: (-3.0) - (-4.5) = 1.5, squared = 2.25
        assert len(arch_errors) == 1
        assert abs(arch_errors[0] - 2.25) < 1e-6


class TestMeanPairwiseErrorOfGroup:
    """Tests for the mean_pairwise_error_of_group function."""
    
    def test_mean_calculation(self):
        """Test calculation of mean pairwise error."""
        # Create a sample DataFrame with a column of values
        df = pd.DataFrame({"value": [1.0, 3.0, 6.0]})
        
        # Calculate mean pairwise error
        result = mean_pairwise_error_of_group(df, "value")
        
        # Check results - should be mean of [(1-3)^2, (1-6)^2, (3-6)^2] = [4, 25, 9]
        expected = (4 + 25 + 9) / 3
        assert abs(result - expected) < 1e-6
        
    def test_with_nan_values(self):
        """Test handling of NaN values."""
        # Create a sample DataFrame with some NaN values
        df = pd.DataFrame({"value": [1.0, np.nan, 5.0]})
        
        # Calculate mean pairwise error, NaNs should be dropped
        result = mean_pairwise_error_of_group(df, "value")
        
        # Check results - should be (1-5)^2 = 16
        assert result == 16.0
        
    def test_with_single_value(self):
        """Test with only one valid value (should return nan)."""
        # Create a sample DataFrame with only one value
        df = pd.DataFrame({"value": [1.0]})
        
        # Calculate mean pairwise error, should return NaN as there are no pairs
        result = mean_pairwise_error_of_group(df, "value")
        
        # Check results
        assert np.isnan(result)
        
    def test_with_no_values(self):
        """Test with no valid values (should return nan)."""
        # Create an empty DataFrame
        df = pd.DataFrame({"value": []})
        
        # Calculate mean pairwise error, should return NaN
        result = mean_pairwise_error_of_group(df, "value")
        
        # Check results
        assert np.isnan(result)


class TestPairwiseBCEOfGroup:
    """Tests for the pairwise_bce_of_group function."""
    
    def test_basic_calculation(self):
        """Test basic BCE calculation."""
        # Create a sample DataFrame with known values
        df = pd.DataFrame({
            "log_prior": [-2.0, -3.0],
            "log_likelihood": [-1.0, -1.5],
            "log_posterior": [-1.5, -2.0]
        })
        
        # Calculate pairwise BCE
        result = pairwise_bce_of_group(df, "log_prior", "log_likelihood", "log_posterior")
        
        # Check evidence estimates: 
        # First row: -2.0 + (-1.0) - (-1.5) = -1.5
        # Second row: -3.0 + (-1.5) - (-2.0) = -2.5
        # Pairwise BCE (squared): (-1.5 - (-2.5))^2 = 1.0
        assert len(result) == 1
        assert abs(result[0] - 1.0) < 1e-6
        
    def test_with_absolute_difference(self):
        """Test BCE calculation with absolute differences (square=False)."""
        # Create a sample DataFrame with known values
        df = pd.DataFrame({
            "log_prior": [-2.0, -3.0],
            "log_likelihood": [-1.0, -1.5],
            "log_posterior": [-1.5, -2.0]
        })
        
        # Calculate pairwise BCE with absolute differences
        result = pairwise_bce_of_group(
            df, "log_prior", "log_likelihood", "log_posterior", square=False
        )
        
        # Check evidence estimates (same as above):
        # First row: -2.0 + (-1.0) - (-1.5) = -1.5
        # Second row: -3.0 + (-1.5) - (-2.0) = -2.5
        # Pairwise BCE (absolute): |-1.5 - (-2.5)| = 1.0
        assert len(result) == 1
        assert abs(result[0] - 1.0) < 1e-6
        
    def test_with_test_logprobs(self, test_logprobs_df):
        """Test BCE calculation with the test logprobs data."""
        # Group by class_type and evidence_text and apply function
        for (class_type, evidence_text), group in test_logprobs_df.groupby(["class_type", "evidence_text"]):
            # Call with square=True to get squared differences
            result = pairwise_bce_of_group(
                group, "prior_logprob", "likelihood_logprob", "posterior_logprob", square=True
            )

            if class_type == "writing_style":
                # BCE between -4.0 and -5.0: (1.0)^2 = 1.0
                assert abs(result[0] - 1.0) < 1e-6
            elif class_type == "architectural_styles":
                # BCE between -3.0 and -4.5: (1.5)^2 = 2.25
                assert abs(result[0] - 2.25) < 1e-6