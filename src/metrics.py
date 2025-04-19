import pandas as pd
import numpy as np
from itertools import combinations
from typing import Optional

def single_evidence_estimate(group: pd.DataFrame, log_prior_col: str, log_likelihood_col: str, log_posterior_col: str) -> Optional[float]:
    return group[log_prior_col] + group[log_likelihood_col] - group[log_posterior_col]

def pairwise_mse_of_group(group: pd.DataFrame, value_col: str) -> Optional[float]:
    """Calculates the pairwise squared errors for a column in a group."""
    valid_values = group[value_col].dropna()
    if len(valid_values) < 2:
        return np.nan # MSE requires at least 2 data points for pairs

    if len(valid_values) > 1000:
         print(f"Warning: Calculating pairwise MSE for a large group ({len(valid_values)} items). This might be slow.")

    # Calculate squared differences for all combinations
    sq_diffs = [(v1 - v2)**2 for v1, v2 in combinations(valid_values, 2)]
    return sq_diffs

def mean_pairwise_mse_of_group(group: pd.DataFrame, value_col: str) -> Optional[float]:
    """Calculates the mean pairwise squared errors for a column in a DataFrame group."""
    return np.mean(pairwise_mse_of_group(group, value_col))
