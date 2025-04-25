import pandas as pd
import numpy as np
from itertools import combinations
from typing import Optional

def single_evidence_estimate(group: pd.DataFrame, log_prior_col: str, log_likelihood_col: str, log_posterior_col: str) -> Optional[float]:
    return group[log_prior_col] + group[log_likelihood_col] - group[log_posterior_col]

def pairwise_error_of_group(group: pd.DataFrame, value_col: str, square: bool = True) -> Optional[float]:
    """Calculates the pairwise squared errors for a column in a group."""
    valid_values = group[value_col].dropna()
    if len(valid_values) < 2:
        return np.nan # MSE requires at least 2 data points for pairs

    if len(valid_values) > 1000:
         print(f"Warning: Calculating pairwise MSE for a large group ({len(valid_values)} items). This might be slow.")

    # Calculate squared differences for all combinations
    if square:
        error = [(v1 - v2)**2 for v1, v2 in combinations(valid_values, 2)]
    else:
        error = [abs(v1 - v2) for v1, v2 in combinations(valid_values, 2)]
    return error

def mean_pairwise_error_of_group(group: pd.DataFrame, value_col: str) -> Optional[float]:
    """Calculates the mean pairwise squared errors for a column in a DataFrame group."""
    return np.mean(pairwise_error_of_group(group, value_col))

def pairwise_bce_of_group(group: pd.DataFrame, log_prior_col: str, log_likelihood_col: str, log_posterior_col: str, square: bool = False) -> Optional[float]:
    """
    Calculates the pairwise binary cross entropy errors between evidence estimates in a group.
    
    First computes evidence estimates for each row using single_evidence_estimate,
    then calculates pairwise errors between these estimates.
    """
    # First, calculate evidence estimates for each row
    group['evidence_estimate'] = single_evidence_estimate(group, log_prior_col, log_likelihood_col, log_posterior_col)
    
    # Then calculate pairwise errors between these estimates
    return pairwise_error_of_group(group, 'evidence_estimate', square=square)