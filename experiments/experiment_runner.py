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
from models.llm_interface import LLMInterface
from models.prompt_templates import generate_likelihood_prompt
from analysis.bce_calculations import compute_bce

def generate_prior_prompt_for_candidate(history: str, candidate: str) -> str:
    """
    Generate a prompt to elicit the prior probability for a specific candidate.
    
    Args:
        history (str): The conversation history.
        candidate (str): The candidate class (e.g., "Shakespeare").
    
    Returns:
        str: The prompt for eliciting the prior probability.
    """
    return f"{history}\nBased on our conversation, what is the probability that you are a fan of {candidate}?"

def generate_posterior_prompt_for_candidate(history: str, candidate: str, evidence: str) -> str:
    """
    Generate a prompt to elicit the posterior probability for a specific candidate after evidence.
    
    Args:
        history (str): The conversation history.
        candidate (str): The candidate class (e.g., "Shakespeare").
        evidence (str): The evidence provided.
    
    Returns:
        str: The prompt for eliciting the posterior probability.
    """
    return f"{history}\nAfter hearing \"{evidence}\", what is the probability that you are a fan of {candidate}?"

def run_full_experiment_multi(history: str, candidate_classes: list, evidence: str, llm: LLMInterface):
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
    for candidate1, candidate2 in combinations(candidate_classes, 2):
        # Generate prompts for each candidate.
        prior_prompt_c1 = generate_prior_prompt_for_candidate(history, candidate1)
        prior_prompt_c2 = generate_prior_prompt_for_candidate(history, candidate2)
        
        likelihood_prompt_c1 = generate_likelihood_prompt(history, candidate1, evidence)
        likelihood_prompt_c2 = generate_likelihood_prompt(history, candidate2, evidence)
        
        posterior_prompt_c1 = generate_posterior_prompt_for_candidate(history, candidate1, evidence)
        posterior_prompt_c2 = generate_posterior_prompt_for_candidate(history, candidate2, evidence)
        
        # Compute probabilities.
        # For prior and posterior, the expected text is the candidate's name.
        # For likelihood, the expected text is the evidence.
        prior_prob_c1 = llm.compute_sentence_probability(prior_prompt_c1, candidate1)
        prior_prob_c2 = llm.compute_sentence_probability(prior_prompt_c2, candidate2)
        
        likelihood_prob_c1 = llm.compute_sentence_probability(likelihood_prompt_c1, evidence)
        likelihood_prob_c2 = llm.compute_sentence_probability(likelihood_prompt_c2, evidence)
        
        posterior_prob_c1 = llm.compute_sentence_probability(posterior_prompt_c1, candidate1)
        posterior_prob_c2 = llm.compute_sentence_probability(posterior_prompt_c2, candidate2)
        
        # Compute BCE for the candidate pair.
        bce_value = compute_bce(prior_prob_c1, prior_prob_c2,
                                likelihood_prob_c1, likelihood_prob_c2,
                                posterior_prob_c1, posterior_prob_c2)
        
        # Print results for this pair.
        print(f"\n--- Results for Pair: {candidate1} vs. {candidate2} ---")
        print(f"Prior probability for {candidate1}: {prior_prob_c1:.4e}")
        print(f"Prior probability for {candidate2}: {prior_prob_c2:.4e}")
        print(f"Likelihood for {candidate1}: {likelihood_prob_c1:.4e}")
        print(f"Likelihood for {candidate2}: {likelihood_prob_c2:.4e}")
        print(f"Posterior probability for {candidate1}: {posterior_prob_c1:.4e}")
        print(f"Posterior probability for {candidate2}: {posterior_prob_c2:.4e}")
        print(f"Bayesian Consistency Error (BCE): {bce_value:.4e}")
        
        # Store the results.
        results[(candidate1, candidate2)] = {
            "prior_prob_c1": prior_prob_c1,
            "prior_prob_c2": prior_prob_c2,
            "likelihood_prob_c1": likelihood_prob_c1,
            "likelihood_prob_c2": likelihood_prob_c2,
            "posterior_prob_c1": posterior_prob_c1,
            "posterior_prob_c2": posterior_prob_c2,
            "BCE": bce_value,
            "prompts": {
                "prior_c1": prior_prompt_c1,
                "prior_c2": prior_prompt_c2,
                "likelihood_c1": likelihood_prompt_c1,
                "likelihood_c2": likelihood_prompt_c2,
                "posterior_c1": posterior_prompt_c1,
                "posterior_c2": posterior_prompt_c2,
            }
        }
        
    return results

if __name__ == "__main__":
    # Example experimental configuration.
    conversation_history = "We've been discussing literary styles and historical contexts in literature."
    candidate_classes = ["Shakespeare", "Mark Twain", "Oscar Wilde", "Charles Dickens"]
    evidence_text = "To thine own self be true."

    # Initialize the LLM interface.
    # For testing, we use the local backend with a lightweight model (e.g., GPT-2).
    llm = LLMInterface(model_name="gpt2", backend="local")

    # Run the experiment across all candidate pairs.
    experiment_results = run_full_experiment_multi(conversation_history, candidate_classes, evidence_text, llm)

    # (Optional) Save or further process experiment_results.
