"""
experiment_runner.py

This script runs an end-to-end experiment for evaluating Bayesian consistency 
in LLMs. For a given conversation history, a list of candidate classes (e.g., authors such as
"Mark Twain", "Oscar Wilde", "Charles Dickens"), and a list of evidence texts, the experiment does the following:

1. Generates prompts to elicit:
   - Prior probabilities (for each candidate) based on conversation history.
   - Likelihood probabilities (for each candidate) given the evidence.
   - Posterior probabilities (for each candidate) after the evidence is presented.
2. Uses the LLM interface to obtain token-level outputs and computes the overall 
   sentence probability for each prompt by iteratively querying the model.
3. Computes the BCE for each pair of candidates using the formula:
       BCE = | log(P(c1|E,H)/P(c2|E,H)) - [ log(P(E|c1,H)/P(E|c2,H)) + log(P(c1|H)/P(c2|H) ) ] |
4. Logs the results to a CSV file (if a log_filepath is provided) and returns the results as a pandas DataFrame.

This implementation uses a dual approach:
- Custom prompt strings for prior and posterior experiments.
- A pre-defined likelihood prompt template for likelihood estimation.
"""

import math
from itertools import combinations
import pandas as pd
import logging

from models.llm_interface import LLMInterface
from src.models.prompt_templates import generate_likelihood_prompt
from src.analysis.bce_calculations import compute_bce

# Setup logging if not already configured
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def run_full_experiment_multi(
    history: str,
    candidate_classes: list,
    evidence_list: list,
    class_elicitation: str,
    evidence_elicitation: str,
    llm: LLMInterface,
    log_filepath: str = None
) -> pd.DataFrame:
    """
    Runs the full experimental pipeline across multiple candidate classes.
    
    For each unique pair of candidates (and for each evidence text in evidence_list), the function:
      1. Generates prompts for prior, likelihood, and posterior probability estimation.
      2. Computes the probability of generating the expected text (candidate name for prior/posterior; evidence for likelihood)
         using the iterative compute_sentence_probability method.
      3. Computes the BCE for the candidate pair.
      4. Logs the results (if log_filepath is provided) and returns a DataFrame.
    
    Args:
        history (str): The conversation history.
        candidate_classes (list): A list of candidate class names (e.g., authors).
        evidence_list (list): A list of evidence texts.
        class_elicitation (str): Text to be appended to the conversation history for candidate elicitation.
        evidence_elicitation (str): Text to be appended for evidence elicitation.
        llm (LLMInterface): An instance of LLMInterface for querying the model.
        log_filepath (str, optional): Path to save the results CSV. If None, results are not saved.
    
    Returns:
        pd.DataFrame: A DataFrame with each row corresponding to the results for a candidate pair with a given evidence text.
    """
    rows = []
    logging.info("Starting experiment over candidate pairs...")

    # Iterate over all unique pairs of candidate classes.
    for class1, class2 in combinations(candidate_classes, 2):
        for evidence in evidence_list:
            # Generate prompts for each candidate.
            prior_prompt = history + class_elicitation
            likelihood_prompt_c1 = history + class_elicitation + class1 + evidence_elicitation
            likelihood_prompt_c2 = history + class_elicitation + class2 + evidence_elicitation
            posterior_prompt = history + evidence_elicitation + evidence + class_elicitation

            logging.info("Processing pair: %s vs. %s with evidence: %s", class1, class2, evidence)

            # Compute probabilities.
            # For prior and posterior, the expected text is the candidate's name.
            # For likelihood, the expected text is the evidence.
            prior_c1 = llm.compute_sentence_probability(prior_prompt, class1)
            prior_c2 = llm.compute_sentence_probability(prior_prompt, class2)

            evidence_likelihood_c1 = llm.compute_sentence_probability(likelihood_prompt_c1, evidence)
            evidence_likelihood_c2 = llm.compute_sentence_probability(likelihood_prompt_c2, evidence)

            posterior_c1 = llm.compute_sentence_probability(posterior_prompt, class1)
            posterior_c2 = llm.compute_sentence_probability(posterior_prompt, class2)

            # Compute BCE for the candidate pair.
            bce_value = compute_bce(
                prior_c1,
                prior_c2,
                evidence_likelihood_c1,
                evidence_likelihood_c2,
                posterior_c1,
                posterior_c2,
            )

            # Log the detailed results for this candidate pair.
            logging.info("Pair %s vs. %s: Prior Ratio = %.4e, Likelihood Ratio = %.4e, Posterior Ratio = %.4e, BCE = %.4e",
                         class1, class2,
                         prior_c1 / prior_c2 if prior_c2 != 0 else float('nan'),
                         evidence_likelihood_c1 / evidence_likelihood_c2 if evidence_likelihood_c2 != 0 else float('nan'),
                         posterior_c1 / posterior_c2 if posterior_c2 != 0 else float('nan'),
                         bce_value)

            # Create a row for this candidate pair.
            row = {
                "class1": class1,
                "class2": class2,
                "evidence": evidence,
                "prior_c1": prior_c1,
                "prior_c2": prior_c2,
                "prior_ratio": prior_c1 / prior_c2 if prior_c2 != 0 else None,
                "likelihood_c1": evidence_likelihood_c1,
                "likelihood_c2": evidence_likelihood_c2,
                "likelihood_ratio": evidence_likelihood_c1 / evidence_likelihood_c2 if evidence_likelihood_c2 != 0 else None,
                "posterior_c1": posterior_c1,
                "posterior_c2": posterior_c2,
                "posterior_ratio": posterior_c1 / posterior_c2 if posterior_c2 != 0 else None,
                "BCE": bce_value,
                "prior_prompt": prior_prompt,
                "likelihood_prompt_c1": likelihood_prompt_c1,
                "likelihood_prompt_c2": likelihood_prompt_c2,
                "posterior_prompt": posterior_prompt,
            }
            rows.append(row)

    # Convert rows into a pandas DataFrame.
    df = pd.DataFrame(rows)

    # If a log filepath is provided, save the DataFrame as CSV.
    if log_filepath:
        try:
            df.to_csv(log_filepath, index=False)
            logging.info("Experiment results saved to %s", log_filepath)
        except Exception as e:
            logging.error("Failed to save experiment results to %s: %s", log_filepath, e)

    return df

if __name__ == "__main__":
    # Example experimental configuration.
    conversation_history = "We've been discussing literary styles and historical contexts in literature."
    candidate_classes = [
        "Shakespeare.",
        "Mark Twain.",
        "Oscar Wilde.",
        "Charles Dickens."
    ]
    evidence_list = [
        " works that bring out the contemporary social conventions and mores of its time rather than focusing on poetic richness and dramatic performance."
    ]
    class_elicitation = " My favourite author is"
    evidence_elicitation = " I prefer reading"

    # Initialize the LLM interface.
    # For testing, we use the local backend with a lightweight model (e.g., GPT-2).
    llm = LLMInterface(model_name="gpt2", backend="local")

    # Optional: specify a filepath to log the results.
    log_filepath = "results/experiment_results.csv"

    # Run the experiment across all candidate pairs.
    experiment_results_df = run_full_experiment_multi(
        conversation_history,
        candidate_classes,
        evidence_list,
        class_elicitation,
        evidence_elicitation,
        llm,
        log_filepath=log_filepath
    )

    # Display the DataFrame.
    print("\n=== Experiment Results DataFrame ===")
    print(experiment_results_df)
