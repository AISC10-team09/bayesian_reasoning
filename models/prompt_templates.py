"""
This module defines a set of predefined prompt templates and utilities for dynamic prompt generation.
It includes templates for eliciting:
  - Prior probabilities (before new evidence is presented)
  - Likelihood probabilities (given a candidate class and evidence)
  - Posterior probabilities (after new evidence is presented)

These functions can randomly select a prompt variation (or use a specified one) to help mitigate biases
introduced by a single prompt formulation.
"""

import random
from typing import List


def generate_prior_prompt(history: str, class1: str, class2: str, variation: int = None) -> str:
    """
    Generates a prompt for eliciting prior probability estimates based on conversation history.

    Args:
        history (str): The conversation history or context.
        class1 (str): The first candidate class.
        class2 (str): The second candidate class.
        variation (int, optional): An index to select a specific prompt variation. If None, a random variation is chosen.

    Returns:
        str: The generated prompt.
    """
    templates = [
        f"{history}\nBased on our conversation so far, do you think I lean more towards being a fan of {class1} or {class2}?",
        f"{history}\nConsidering what we've discussed, which is more likely: that I favor {class1} or {class2}?",
        f"{history}\nGiven our discussion, would you say I have a preference for {class1} or for {class2}?",
    ]
    if variation is not None:
        return templates[variation % len(templates)]
    else:
        return random.choice(templates)


def generate_likelihood_prompt(history: str, class_label: str, evidence: str, variation: int = None) -> str:
    """
    Generates a prompt for eliciting the likelihood of a particular piece of evidence
    given a candidate class.

    Args:
        history (str): The conversation history or context.
        class_label (str): The candidate class (e.g., "Shakespeare" or "Mark Twain").
        evidence (str): The evidence or text snippet to evaluate.
        variation (int, optional): An index to select a specific prompt variation. If None, a random variation is chosen.

    Returns:
        str: The generated prompt.
    """
    templates = [
        f"{history}\nAssuming you are a fan of {class_label}, how likely is it that you would say: \"{evidence}\"?",
        f"{history}\nIf you identify as a fan of {class_label}, what is the probability you would express: \"{evidence}\"?",
        f"{history}\nGiven your affinity for {class_label}, how probable is it that you would articulate: \"{evidence}\"?",
    ]
    if variation is not None:
        return templates[variation % len(templates)]
    else:
        return random.choice(templates)


def generate_posterior_prompt(history: str, evidence: str, class1: str, class2: str, variation: int = None) -> str:
    """
    Generates a prompt for eliciting posterior probability estimates after new evidence is provided.

    Args:
        history (str): The conversation history or context.
        evidence (str): The piece of evidence that has been presented.
        class1 (str): The first candidate class.
        class2 (str): The second candidate class.
        variation (int, optional): An index to select a specific prompt variation. If None, a random variation is chosen.

    Returns:
        str: The generated prompt.
    """
    templates = [
        f"{history}\nNow that you've heard: \"{evidence}\", which is more likely: that I'm a fan of {class1} or of {class2}?",
        f"{history}\nAfter considering the evidence \"{evidence}\", do you think I lean towards {class1} or {class2}?",
        f"{history}\nGiven the new information: \"{evidence}\", who seems more likely: a fan of {class1} or a fan of {class2}?",
    ]
    if variation is not None:
        return templates[variation % len(templates)]
    else:
        return random.choice(templates)


def generate_all_prompts(history: str, class1: str, class2: str, evidence: str) -> List[str]:
    """
    Generates a set of prompts covering prior, likelihood (for both classes), and posterior elicitation.

    Args:
        history (str): The conversation history or context.
        class1 (str): The first candidate class.
        class2 (str): The second candidate class.
        evidence (str): The evidence to be used for likelihood and posterior prompts.

    Returns:
        List[str]: A list containing:
            - Prior prompt
            - Likelihood prompt for class1
            - Likelihood prompt for class2
            - Posterior prompt
    """
    prior_prompt = generate_prior_prompt(history, class1, class2)
    likelihood_prompt_class1 = generate_likelihood_prompt(history, class1, evidence)
    likelihood_prompt_class2 = generate_likelihood_prompt(history, class2, evidence)
    posterior_prompt = generate_posterior_prompt(history, evidence, class1, class2)
    return [prior_prompt, likelihood_prompt_class1, likelihood_prompt_class2, posterior_prompt]


if __name__ == "__main__":
    # Example usage for testing prompt generation.
    test_history = "We've been discussing our literary preferences and writing styles."
    test_class1 = "Shakespeare"
    test_class2 = "Mark Twain"
    test_evidence = "To thine own self be true."

    print("Prior Prompt:")
    print(generate_prior_prompt(test_history, test_class1, test_class2))
    print("\nLikelihood Prompt (for Shakespeare):")
    print(generate_likelihood_prompt(test_history, test_class1, test_evidence))
    print("\nLikelihood Prompt (for Mark Twain):")
    print(generate_likelihood_prompt(test_history, test_class2, test_evidence))
    print("\nPosterior Prompt:")
    print(generate_posterior_prompt(test_history, test_evidence, test_class1, test_class2))

    print("\nAll Prompts:")
    for prompt in generate_all_prompts(test_history, test_class1, test_class2, test_evidence):
        print(prompt)
