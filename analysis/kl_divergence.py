"""
This module provides functions to compute the Kullback-Leibler (KL) divergence
between two discrete probability distributions.

For two distributions P and Q defined over the same set, the KL divergence is given by:
    KL(P || Q) = sum_i P(i) * log(P(i) / Q(i))

A small epsilon is used to ensure numerical stability and to avoid division by zero.
"""

import math
from typing import List


def compute_kl_divergence(p: List[float], q: List[float], epsilon: float = 1e-12) -> float:
    """
    Computes the KL divergence KL(P || Q) for two discrete probability distributions.

    Args:
        p (List[float]): The true probability distribution P.
        q (List[float]): The approximating probability distribution Q.
        epsilon (float, optional): A small constant to avoid division by zero. Default is 1e-12.

    Returns:
        float: The computed KL divergence.

    Raises:
        ValueError: If the input distributions are not of the same length.
    """
    if len(p) != len(q):
        raise ValueError("The two distributions must have the same length.")

    kl_div = 0.0
    for pi, qi in zip(p, q):
        # Ensure both probabilities are at least epsilon
        safe_pi = max(pi, epsilon)
        safe_qi = max(qi, epsilon)
        kl_div += safe_pi * math.log(safe_pi / safe_qi)
    return kl_div


# Example usage:
if __name__ == "__main__":
    # Example distributions for a transparent class.
    # Here, P might be the true likelihood distribution and Q the LLM-assigned likelihood.
    p = [0.7, 0.2, 0.1]  # True distribution
    q = [0.6, 0.25, 0.15]  # LLM-assigned distribution

    kl_value = compute_kl_divergence(p, q)
    print("KL Divergence (KL(P || Q)):", kl_value)
