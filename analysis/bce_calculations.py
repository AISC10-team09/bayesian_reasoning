"""
This module provides functions to compute the Bayesian Consistency Error (BCE)
given extracted probability estimates from an LLM.

Bayes' rule in ratio form implies:
    log(P(c1|E,H)/P(c2|E,H)) = log(P(E|c1,H)/P(E|c2,H)) + log(P(c1|H)/P(c2|H))

The BCE is defined as the absolute difference between the LHS and RHS:
    BCE = | log(P(c1|E,H)/P(c2|E,H)) - [log(P(E|c1,H)/P(E|c2,H)) + log(P(c1|H)/P(c2|H))] |

Functions:
    - compute_log_ratio: Computes log(prob1/prob2) safely.
    - compute_bce: Computes the BCE given the prior, likelihood, and posterior probabilities.
"""

import math


def compute_log_ratio(prob1: float, prob2: float, epsilon: float = 1e-12) -> float:
    """
    Computes the logarithmic ratio log(prob1/prob2) in a numerically stable way.

    Args:
        prob1 (float): The first probability.
        prob2 (float): The second probability.
        epsilon (float, optional): A small constant to avoid division by zero. Default is 1e-12.

    Returns:
        float: The computed logarithmic ratio.
    """
    # Ensure probabilities are non-zero by using a small epsilon.
    safe_prob1 = max(prob1, epsilon)
    safe_prob2 = max(prob2, epsilon)
    return math.log(safe_prob1) - math.log(safe_prob2)


def compute_bce(prior_prob_c1: float, prior_prob_c2: float,
                likelihood_c1: float, likelihood_c2: float,
                posterior_c1: float, posterior_c2: float) -> float:
    """
    Computes the Bayesian Consistency Error (BCE) for a given set of probability estimates.

    The BCE is defined as:
        BCE = | log(P(c1|E,H)/P(c2|E,H)) - [ log(P(E|c1,H)/P(E|c2,H)) + log(P(c1|H)/P(c2|H) ) ] |

    Args:
        prior_prob_c1 (float): Prior probability P(c1 | H) for candidate class 1.
        prior_prob_c2 (float): Prior probability P(c2 | H) for candidate class 2.
        likelihood_c1 (float): Likelihood P(E | c1, H) for candidate class 1.
        likelihood_c2 (float): Likelihood P(E | c2, H) for candidate class 2.
        posterior_c1 (float): Posterior probability P(c1 | E, H) for candidate class 1.
        posterior_c2 (float): Posterior probability P(c2 | E, H) for candidate class 2.

    Returns:
        float: The absolute error (BCE) between the LHS and the RHS of Bayes' rule in log space.
    """
    # Compute the log ratios for each component
    log_prior_ratio = compute_log_ratio(prior_prob_c1, prior_prob_c2)
    log_likelihood_ratio = compute_log_ratio(likelihood_c1, likelihood_c2)
    log_posterior_ratio = compute_log_ratio(posterior_c1, posterior_c2)

    # Calculate BCE as the absolute difference
    bce = abs(log_posterior_ratio - (log_likelihood_ratio + log_prior_ratio))
    return bce


# Example usage:
if __name__ == "__main__":
    # Example probabilities for candidate classes c1 and c2:
    prior_c1 = 0.6  # P(c1 | H)
    prior_c2 = 0.4  # P(c2 | H)
    likelihood_c1 = 0.7  # P(E | c1, H)
    likelihood_c2 = 0.3  # P(E | c2, H)
    posterior_c1 = 0.8  # P(c1 | E, H)
    posterior_c2 = 0.2  # P(c2 | E, H)

    bce_value = compute_bce(prior_c1, prior_c2, likelihood_c1, likelihood_c2, posterior_c1, posterior_c2)
    print("Bayesian Consistency Error (BCE):", bce_value)
