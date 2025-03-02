"""
This script runs an end-to-end experiment for evaluating Bayesian consistency
in LLMs. For a given conversation history, two candidate classes (e.g., "Shakespeare"
and "Mark Twain"), and a piece of evidence, the experiment does the following:

1. Generates prompts to elicit:
   - Prior probabilities (for each candidate) based on conversation history.
   - Likelihood probabilities (for each candidate) given the evidence.
   - Posterior probabilities (for each candidate) after the evidence is presented.
2. Uses the LLM interface to obtain token-level outputs and computes the overall
   sentence probability for each prompt.
3. Computes the BCE using the formula:
       BCE = | log(P(c1|E,H)/P(c2|E,H)) - ( log(P(E|c1,H)/P(E|c2,H)) + log(P(c1|H)/P(c2|H) ) ) |
4. Prints (and returns) a summary of the results.

This implementation uses a dual approach: custom prompt strings for prior and
posterior experiments and the pre-defined likelihood prompt template for likelihood.
"""

import math
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


def run_full_experiment(history: str, candidate1: str, candidate2: str, evidence: str, llm: LLMInterface):
    """
    Runs the full experimental pipeline:
        - Elicits prior, likelihood, and posterior probabilities for two candidate classes.
        - Computes the BCE based on these probabilities.

    Args:
        history (str): Conversation history.
        candidate1 (str): First candidate class.
        candidate2 (str): Second candidate class.
        evidence (str): Evidence to be considered.
        llm (LLMInterface): An instance of LLMInterface for querying the model.

    Returns:
        dict: A dictionary containing the computed probabilities and BCE, as well as the prompts used.
    """
    # Generate prior prompts for each candidate.
    prior_prompt_c1 = generate_prior_prompt_for_candidate(history, candidate1)
    prior_prompt_c2 = generate_prior_prompt_for_candidate(history, candidate2)

    # Generate likelihood prompts using the pre-defined template.
    likelihood_prompt_c1 = generate_likelihood_prompt(history, candidate1, evidence)
    likelihood_prompt_c2 = generate_likelihood_prompt(history, candidate2, evidence)

    # Generate posterior prompts for each candidate.
    posterior_prompt_c1 = generate_posterior_prompt_for_candidate(history, candidate1, evidence)
    posterior_prompt_c2 = generate_posterior_prompt_for_candidate(history, candidate2, evidence)

    # Query the LLM for each prompt.
    print("Querying LLM for prior probabilities...")
    # prior_response_c1 = llm.get_output_probabilities(prior_prompt_c1)
    # prior_response_c2 = llm.get_output_probabilities(prior_prompt_c2)
    prior_response_c1 = llm.get_output_probabilities(prior_prompt_c1, candidate1)
    prior_response_c2 = llm.get_output_probabilities(prior_prompt_c2, candidate2)

    print("Querying LLM for likelihood probabilities...")
    # likelihood_response_c1 = llm.get_output_probabilities(likelihood_prompt_c1)
    # likelihood_response_c2 = llm.get_output_probabilities(likelihood_prompt_c2)
    likelihood_response_c1 = llm.get_output_probabilities(likelihood_prompt_c1, evidence)
    likelihood_response_c2 = llm.get_output_probabilities(likelihood_prompt_c2, evidence)

    # print("Querying LLM for posterior probabilities...")
    # posterior_response_c1 = llm.get_output_probabilities(posterior_prompt_c1)
    # posterior_response_c2 = llm.get_output_probabilities(posterior_prompt_c2)
    posterior_response_c1 = llm.get_output_probabilities(posterior_prompt_c1, candidate1)
    posterior_response_c2 = llm.get_output_probabilities(posterior_prompt_c2, candidate2)

    # Compute overall sentence probabilities from token-level log probabilities.
    prior_prob_c1 = llm.compute_sentence_probability(prior_response_c1.get("token_logprobs", []))
    prior_prob_c2 = llm.compute_sentence_probability(prior_response_c2.get("token_logprobs", []))
    likelihood_c1 = llm.compute_sentence_probability(likelihood_response_c1.get("token_logprobs", []))
    likelihood_c2 = llm.compute_sentence_probability(likelihood_response_c2.get("token_logprobs", []))
    posterior_prob_c1 = llm.compute_sentence_probability(posterior_response_c1.get("token_logprobs", []))
    posterior_prob_c2 = llm.compute_sentence_probability(posterior_response_c2.get("token_logprobs", []))

    # Compute BCE using the formula:
    # BCE = | log(P(c1|E,H)/P(c2|E,H)) - [ log(P(E|c1,H)/P(E|c2,H)) + log(P(c1|H)/P(c2|H) ) ] |
    bce_value = compute_bce(prior_prob_c1, prior_prob_c2,
                             likelihood_c1, likelihood_c2,
                             posterior_prob_c1, posterior_prob_c2)

    # Print and return the results.
    print("\n--- Experiment Results ---")
    print(f"Prior probability for {candidate1}: {prior_prob_c1:.4e}")
    print(f"Prior probability for {candidate2}: {prior_prob_c2:.4e}")
    print(f"Likelihood for {candidate1}: {likelihood_c1:.4e}")
    print(f"Likelihood for {candidate2}: {likelihood_c2:.4e}")
    print(f"Posterior probability for {candidate1}: {posterior_prob_c1:.4e}")
    print(f"Posterior probability for {candidate2}: {posterior_prob_c2:.4e}")
    print(f"Bayesian Consistency Error (BCE): {bce_value:.4e}")

    results = {
        "prior_prob_c1": prior_prob_c1,
        "prior_prob_c2": prior_prob_c2,
        "likelihood_c1": likelihood_c1,
        "likelihood_c2": likelihood_c2,
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
    candidate_class1 = "Shakespeare"
    candidate_class2 = "Mark Twain"
    evidence_text = "To thine own self be true."

    # Initialize the LLM interface.
    # For testing, we can use the local backend with a lightweight model such as GPT-2.
    # Replace 'gpt2' with your chosen model name and provide an API key if using the OpenAI backend.
    llm = LLMInterface(model_name="gpt2", backend="local")

    # Run the full experiment.
    experiment_results = run_full_experiment(conversation_history, candidate_class1, candidate_class2, evidence_text,
                                             llm)

    # (Optional) Save the results to a file or further process them.
    # For example, you could serialize experiment_results as JSON.
