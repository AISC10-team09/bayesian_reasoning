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

# import math

from models.llm_interface_2 import LLMInterface_2
from analysis.bce_calculations import compute_bce


def generate_prior_prompt(history: str, prompt_prior: str) -> str:
    """
    generate a prompt to elicit the prior probability for a specific candidate class
    
    args:
        history (str): the conversation history
        prompt_prior (str): the prompt
    
    teturns:
        str: the prompt for eliciting the prior probability
    """
    return f"{history}\n{prompt_prior}"


def generate_likelihood_prompt_1(history: str, evidence: str, prompt_evidence: str) -> str:
    """
    generates a prompt for eliciting the likelihood of a particular piece of evidence given a conversation history and a
    candidate class.

    args:
        history (str): the conversation history
        evidence (str): the evidence to evaluate

    returns:
        str: The generated prompt.
    """

    return f"{history}\n{evidence}\n{prompt_evidence}"


def generate_posterior_prompt(history: str,  evidence: str, prompt_posterior: str) -> str:
    """
    generate a prompt to elicit the posterior probability for a specific candidate class
    
    Args:
        history (str): the conversation history
        prompt_posterior (str): the prompt
        evidence (str): the evidence provided
    
    Returns:
        str: the prompt for eliciting the posterior probability
    """
    return f"{history}\nAfter considering that {evidence}, {prompt_posterior}"



def run_full_experiment_multi(
        history: str,
        class_labels: str,
        prompt_prior: str,
        evidence: str,
        prompt_evidence: str,
        expected_text: str,
        prompt_posterior: str,
        llm: LLMInterface_2
    ):
    """
    runs the full experiment pipeline
    
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

    print("class labels:", class_labels, end = "\n\n")    
        
    # generate the prompts
    prior_prompt = generate_prior_prompt(history, prompt_prior)        
    likelihood_prompt = generate_likelihood_prompt_1(history, evidence, prompt_evidence)
    posterior_prompt = generate_posterior_prompt(history, evidence, prompt_posterior)
               
    # compute probabilities.
    # for prior and posterior, the expected text is the class label
    # For likelihood, the expected text is the evidence.
    print("prior:", prior_prompt, end = "\n\n")
    prior_prob_c1 = llm.compute_probability(prior_prompt, class_labels[0])
    prior_prob_c2 = llm.compute_probability(prior_prompt, class_labels[1])
    
    print("likelihood:", likelihood_prompt, end = "\n\n")
    likelihood_prob_c1 = llm.compute_probability(likelihood_prompt, evidence)
    likelihood_prob_c2 = llm.compute_probability(likelihood_prompt, evidence)

    print("posterior:", posterior_prompt, end = "\n\n")
    posterior_prob_c1 = llm.compute_probability(posterior_prompt, class_labels[0])
    posterior_prob_c2 = llm.compute_probability(posterior_prompt, class_labels[1])
    
    # compute bayesian coherence error for the class pairs
    bce_value = compute_bce(prior_prob_c1, prior_prob_c2,
                            likelihood_prob_c1, likelihood_prob_c2,
                            posterior_prob_c1, posterior_prob_c2)
    
    # print results for this pair.
    print(f"\n--- Results for Pair: {class_labels[0]} vs. {class_labels[1]} ---")
    print(f"Prior probability for {class_labels[0]}: {prior_prob_c1:.4e}")
    print(f"Prior probability for {class_labels[1]}: {prior_prob_c2:.4e}")
    print(f"Likelihood for {class_labels[0]}: {likelihood_prob_c1:.4e}")
    print(f"Likelihood for {class_labels[1]}: {likelihood_prob_c2:.4e}")
    print(f"Posterior probability for {class_labels[0]}: {posterior_prob_c1:.4e}")
    print(f"Posterior probability for {class_labels[1]}: {posterior_prob_c2:.4e}")
    print(f"Bayesian Consistency Error (BCE): {bce_value:.4e}")
    
    # store the results.
    results[(class_labels[0], class_labels[1])] = {
        "prior_prob_c1": prior_prob_c1,
        "prior_prob_c2": prior_prob_c2,
        "likelihood_prob_c1": likelihood_prob_c1,
        "likelihood_prob_c2": likelihood_prob_c2,
        "posterior_prob_c1": posterior_prob_c1,
        "posterior_prob_c2": posterior_prob_c2,
        "BCE": bce_value,
        "prompts": {
            "prior": prior_prompt,
            "likelihood": likelihood_prompt,
            "posterior": posterior_prompt,
        }
    }
    
    return results


if __name__ == "__main__":

    # example experimental configuration
    conversation_history = "We've been discussing literary styles and historical contexts in literature."
    class_labels = ["Shakespeare", "Charles Dickens"]
    prompt_prior = " I enjoy reading books written by English authors."
    prompt_posterior = "Who do you think my favourite author is?"
    evidence = "I like works that bring out the contemporary social conventions and mores of its time rather than" \
    " focusing on poetic richness and dramatic performance."
    expected_text = "You prefer reading about lives of the working class"
    prompt_evidence = "What kinds of stories do you think I prefer reading?"

    # initialize the llm interface.
    # For testing, we use the local backend with a lightweight model (e.g., GPT-2).
    llm = LLMInterface_2(model_name = "gpt2", backend = "local")

    # run the experiment
    experiment_results = run_full_experiment_multi(
        conversation_history,
        class_labels,
        prompt_prior,
        evidence,
        prompt_evidence,
        expected_text,
        prompt_posterior,
        llm
    )