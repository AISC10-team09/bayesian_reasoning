import pandas as pd
import numpy as np
from itertools import combinations
from typing import Optional

def calculate_bce_sum(df: pd.DataFrame,
                      prior_col='prior_logprob',
                      likelihood_col='likelihood_logprob',
                      posterior_col='posterior_logprob') -> pd.Series:
    """Calculates log_prior + log_likelihood - log_posterior element-wise."""
    if not all(col in df.columns for col in [prior_col, likelihood_col, posterior_col]):
        raise ValueError(f"Missing required columns for BCE sum: Need {prior_col}, {likelihood_col}, {posterior_col}")
    # Ensure numeric types, coerce errors to NaN
    prior = pd.to_numeric(df[prior_col], errors='coerce')
    likelihood = pd.to_numeric(df[likelihood_col], errors='coerce')
    posterior = pd.to_numeric(df[posterior_col], errors='coerce')
    return prior + likelihood - posterior

def calculate_variance_on_group(group: pd.DataFrame, value_col: str) -> Optional[float]:
    """Calculates variance of a specific column within a DataFrame group."""
    if group[value_col].count() < 2: # Variance requires at least 2 data points
        return np.nan
    return group[value_col].var()

def calculate_pairwise_mse_on_group(group: pd.DataFrame, value_col: str) -> Optional[float]:
    """Calculates the pairwise squared errors for a column in a group."""
    valid_values = group[value_col].dropna()
    if len(valid_values) < 2:
        return np.nan # MSE requires at least 2 data points for pairs

    if len(valid_values) > 1000:
         print(f"Warning: Calculating pairwise MSE for a large group ({len(valid_values)} items). This might be slow.")

    # Calculate squared differences for all combinations
    sq_diffs = [(v1 - v2)**2 for v1, v2 in combinations(valid_values, 2)]
    return sq_diffs