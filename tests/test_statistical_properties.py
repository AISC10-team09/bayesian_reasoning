"""
Tests to verify that our metrics have the expected statistical properties.
"""
import pytest
pytestmark = [pytest.mark.metrics, pytest.mark.statistical]
import numpy as np
import pandas as pd
from src.metrics import (
    single_evidence_estimate, 
    pairwise_error_of_group, 
    pairwise_bce_of_group
)

class TestStatisticalProperties:
    """Tests for statistical properties of our metrics."""
    
    def test_bce_symmetry_property(self):
        """Test that BCE exhibits symmetry (BCE between A and B equals BCE between B and A)."""
        # Create a test DataFrame with two rows that should be compared
        df = pd.DataFrame({
            "prior_logprob": [-5.0, -6.0],
            "likelihood_logprob": [-3.0, -2.0],
            "posterior_logprob": [-4.0, -3.0]
        })
        
        # Calculate BCE
        bce_values = pairwise_bce_of_group(
            df, "prior_logprob", "likelihood_logprob", "posterior_logprob", square=True
        )
        
        # In theory, we only have one comparison (A vs B)
        assert len(bce_values) == 1, f"Expected 1 BCE value, got {len(bce_values)}"
        
        # Calculate evidence estimates manually for A and B
        evidence_A = df.iloc[0]["prior_logprob"] + df.iloc[0]["likelihood_logprob"] - df.iloc[0]["posterior_logprob"]
        evidence_B = df.iloc[1]["prior_logprob"] + df.iloc[1]["likelihood_logprob"] - df.iloc[1]["posterior_logprob"]
        
        # Verify symmetry - BCE(A,B) = BCE(B,A) = (evidence_A - evidence_B)²
        expected_bce = (evidence_A - evidence_B) ** 2
        assert abs(bce_values[0] - expected_bce) < 1e-6, f"BCE symmetry violated: {bce_values[0]} ≠ {expected_bce}"
        
    def test_bce_non_negativity(self):
        """Test that BCE is always non-negative (≥ 0)."""
        # Create test data with variation in values
        test_cases = [
            # Case 1: Similar evidence estimates
            pd.DataFrame({
                "prior_logprob": [-5.0, -5.1],
                "likelihood_logprob": [-3.0, -3.0],
                "posterior_logprob": [-4.0, -4.1]
            }),
            # Case 2: Very different evidence estimates
            pd.DataFrame({
                "prior_logprob": [-5.0, -10.0],
                "likelihood_logprob": [-3.0, -8.0],
                "posterior_logprob": [-4.0, -9.0]
            }),
            # Case 3: One perfectly consistent case, one inconsistent
            pd.DataFrame({
                "prior_logprob": [-5.0, -6.0],
                "likelihood_logprob": [-3.0, -2.0],
                "posterior_logprob": [-8.0, -8.0]  # P ≠ P(H) + P(E|H) - P(H|E)
            })
        ]
        
        for i, df in enumerate(test_cases):
            # Calculate BCE with square=True (squared differences)
            bce_values = pairwise_bce_of_group(
                df, "prior_logprob", "likelihood_logprob", "posterior_logprob", square=True
            )
            
            # BCE should always be non-negative
            assert all(value >= 0 for value in bce_values), f"Case {i+1}: BCE should be non-negative"
            
            # Calculate BCE with square=False (absolute differences)
            bce_abs_values = pairwise_bce_of_group(
                df, "prior_logprob", "likelihood_logprob", "posterior_logprob", square=False
            )
            
            # Absolute BCE should always be non-negative
            assert all(value >= 0 for value in bce_abs_values), f"Case {i+1}: Absolute BCE should be non-negative"
    
    def test_bce_triangle_inequality(self):
        """Test that BCE approximately respects the triangle inequality.
        
        For metrics that behave like distances, the triangle inequality should hold:
        BCE(A,C) ≤ BCE(A,B) + BCE(B,C)
        """
        # Create test data with three evidence estimates
        df = pd.DataFrame({
            "prior_logprob": [-5.0, -6.0, -7.0],
            "likelihood_logprob": [-3.0, -2.0, -1.0],
            "posterior_logprob": [-4.0, -3.0, -2.0]
        })
        
        # Calculate evidence estimates
        df["evidence_estimate"] = single_evidence_estimate(
            df, "prior_logprob", "likelihood_logprob", "posterior_logprob"
        )
        
        # Calculate pairwise distances (without squaring)
        distances = []
        for i in range(3):
            for j in range(i+1, 3):
                dist = abs(df.iloc[i]["evidence_estimate"] - df.iloc[j]["evidence_estimate"])
                distances.append((i, j, dist))
        
        # Verify triangle inequality for all triplets
        A, B, C = 0, 1, 2  # Indices for the three points
        AB = next(d for i, j, d in distances if (i, j) == (A, B))
        BC = next(d for i, j, d in distances if (i, j) == (B, C))
        AC = next(d for i, j, d in distances if (i, j) == (A, C))
        
        # The triangle inequality: AB + BC >= AC
        assert AB + BC >= AC - 1e-6, f"Triangle inequality violated: {AB} + {BC} < {AC}"