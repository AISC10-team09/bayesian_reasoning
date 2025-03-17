"""
experiment_runner.py

This script runs an end-to-end experiment for evaluating Bayesian consistency
in LLMs. For a given conversation history, a list of candidate classes (e.g., authors such as
"Mark Twain", "Oscar Wilde", "Charles Dickens"), and a piece of evidence, the experiment does the following:

1. Generates prompts to elicit:
   - Prior probabilities (for each candidate) based on conversation history.
   - Likelihood probabilities (for each candidate) given the evidence.
   - Posterior probabilities (for each candidate) after the evidence is presented.
2. Uses the LLM interface to obtain token-level outputs and computes the overall
   sentence probability for each prompt by iteratively querying the model.
3. Computes the BCE for each pair of candidates using the formula:
       BCE = | log(P(c1|E,H)/P(c2|E,H)) - [ log(P(E|c1,H)/P(E|c2,H)) + log(P(c1|H)/P(c2|H) ) ] |
4. Prints (and returns) a summary of the results.

This implementation uses a dual approach:
- Custom prompt strings for prior and posterior experiments.
- A pre-defined likelihood prompt template for likelihood estimation.
"""

import math
from itertools import combinations
from src.models.llm_interface_2 import LLMInterface_2
from src.analysis.bce_calculations import compute_bce


def run_full_experiment_multi(
    history: str,
    candidate_classes: list,
    evidence_list: list,
    class_elicitation: str,
    evidence_elicitation: str,
    llm: LLMInterface_2,
):
    """
    Runs the full experimental pipeline across multiple candidate classes.

    For each pair of candidates in candidate_classes, the function:
      1. Generates prompts for prior, likelihood, and posterior probability estimation.
      2. Computes the probability of generating the expected text (candidate name for prior/posterior; evidence for likelihood)
         using the iterative compute_sentence_probability method.
      3. Computes the BCE for the candidate pair.
      4. Returns a dictionary with results keyed by the candidate pair.

    Args:
        history (str): The conversation history.
        candidate_classes (list): A list of candidate class names (e.g., authors).
        evidence (str): The evidence text.
        llm (LLMInterface): An instance of LLMInterface for querying the model.

    Returns:
        dict: A dictionary where each key is a tuple (candidate1, candidate2) and the value is a dict containing:
              - Prior, likelihood, and posterior probabilities for each candidate.
              - The computed Bayesian Consistency Error (BCE).
              - The prompts used for each candidate.
    """
    results = {}

    # Iterate over all unique pairs of candidate classes.
    for class1, class2 in combinations(candidate_classes, 2):
        for evidence in evidence_list:
            # Generate prompts for each candidate.
            prior_prompt = history + class_elicitation

            likelihood_prompt_c1 = (
                history + class_elicitation + class1 + evidence_elicitation
            )
            likelihood_prompt_c2 = (
                history + class_elicitation + class2 + evidence_elicitation
            )

            posterior_prompt = (
                history + evidence_elicitation + evidence + class_elicitation
            )

            # Compute probabilities.
            # For prior and posterior, the expected text is the candidate's name.
            # For likelihood, the expected text is the evidence.
            prior_c1 = llm.compute_probability(prior_prompt, class1)
            prior_c2 = llm.compute_probability(prior_prompt, class2)

            evidence_likelihood_c1 = llm.compute_probability(
                likelihood_prompt_c1, evidence
            )
            evidence_likelihood_c2 = llm.compute_probability(
                likelihood_prompt_c2, evidence
            )

            posterior_c1 = llm.compute_probability(posterior_prompt, class1)
            posterior_c2 = llm.compute_probability(posterior_prompt, class2)

            # Compute BCE for the candidate pair.
            bce_value = compute_bce(
                prior_c1,
                prior_c2,
                evidence_likelihood_c1,
                evidence_likelihood_c2,
                posterior_c1,
                posterior_c2,
            )

            # Print results for this pair.
            print(f"\n--- Results for Pair: {class1} vs. {class2} ---")
            print(f"Prior probability for {class1}: {prior_c1:.4e}")
            print(f"Prior probability for {class2}: {prior_c2:.4e}")
            print(f"Prior ratio: {prior_c1/prior_c2:.4e}")
            print(f"Likelihood for {class1}: {evidence_likelihood_c1:.4e}")
            print(f"Likelihood for {class2}: {evidence_likelihood_c2:.4e}")
            print(f"Likelihood ratio: {evidence_likelihood_c1/evidence_likelihood_c2:.4e}")
            print(f"Posterior probability for {class1}: {posterior_c1:.4e}")
            print(f"Posterior probability for {class2}: {posterior_c2:.4e}")
            print(f"Posterior ratio: {posterior_c1/posterior_c2:.4e}")
            print(f"Bayesian Consistency Error (BCE): {bce_value:.4e}")

            # Store the results.
            results[(class1, class2, evidence)] = {
                "prior_c1": prior_c1,
                "prior_c2": prior_c2,
                "evidence_likelihood_c1": evidence_likelihood_c1,
                "evidence_likelihood_c2": evidence_likelihood_c2,
                "posterior_c1": posterior_c1,
                "posterior_c2": posterior_c2,
                "BCE": bce_value,
                "prompts": {
                    "prior_prompt": prior_prompt,
                    "likelihood_prompt_c1": likelihood_prompt_c1,
                    "likelihood_prompt_c2": likelihood_prompt_c2,
                    "posterior_prompt": posterior_prompt,
                },
            }

    return results


if __name__ == "__main__":
    # Example experimental configuration.
    conversation_history = (
        "We've been discussing literary styles and historical contexts in literature."
    )
    candidate_classes = [
        " Shakespeare.",
        " Mark Twain.",
        " Oscar Wilde.",
        " Charles Dickens.",
    ]
    evidence_list = [
        " works that bring out the contemporary social conventions and mores of its time rather than"
        " focusing on poetic richness and dramatic performance."
    ]
    class_elicitation = " My favourite author is"
    evidence_elicitation = " I prefer reading"

    # Initialize the LLM interface.
    # For testing, we use the local backend with a lightweight model (e.g., GPT-2).
    llm = LLMInterface_2(model_name="gpt2", backend="local")

    # Run the experiment across all candidate pairs.
    experiment_results = run_full_experiment_multi(
        conversation_history,
        candidate_classes,
        evidence_list,
        class_elicitation,
        evidence_elicitation,
        llm,
    )