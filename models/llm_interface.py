"""
This module provides an interface to interact with large language models
to extract token-level probability estimates. It supports two backends:
- "openai": Uses the OpenAI API.
- "local": Uses a local model from Hugging Face Transformers (e.g., GPT-2).
"""

import math
from typing import List, Dict, Any

# Attempt to import OpenAI library for API-based access.
try:
    import openai
except ImportError:
    openai = None

# Imports for local inference using Hugging Face Transformers.
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class LLMInterface:
    """
    A class to interface with an LLM via either the OpenAI API or a local model.

    Attributes:
        model_name (str): The model identifier (e.g., "text-davinci-003" or "gpt2").
        api_key (str): API key for accessing the OpenAI API (if backend is "openai").
        backend (str): "openai" or "local". Default is "openai".
        max_tokens (int): Maximum number of tokens to generate for each call.
        temperature (float): Sampling temperature for generation.
        logprobs (int): Number of log probabilities to request per token.
    """

    def __init__(self, model_name: str, api_key: str = None, backend: str = "openai",
                 max_tokens: int = 50, temperature: float = 0.7, logprobs: int = 5):
        """
        Initializes the LLMInterface with model settings.

        Args:
            model_name (str): The model identifier.
            api_key (str, optional): API key for the OpenAI API (required for "openai" backend).
            backend (str, optional): Either "openai" or "local". Default is "openai".
            max_tokens (int, optional): Maximum tokens to generate. Default is 50.
            temperature (float, optional): Sampling temperature. Default is 0.7.
            logprobs (int, optional): Number of log probabilities to return per token. Default is 5.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.backend = backend
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.logprobs = logprobs

        if self.backend == "openai":
            if not openai:
                raise ImportError("OpenAI library is not installed. Install it or use backend='local'.")
            if not self.api_key:
                raise ValueError("API key is required for OpenAI backend.")
            openai.api_key = self.api_key
        elif self.backend == "local":
            # For local inference, load the tokenizer and model from Hugging Face.
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self.model.eval()  # Set the model to evaluation mode.
        else:
            raise ValueError("Invalid backend specified. Choose 'openai' or 'local'.")

    def get_output_probabilities(self, prompt: str) -> Dict[str, Any]:
        """
        Sends a prompt to the LLM and retrieves token-level probabilities.

        For "openai", calls the OpenAI API to get token-level log probabilities.
        For "local", uses Hugging Face Transformers to compute log probabilities.

        Args:
            prompt (str): The input prompt.

        Returns:
            dict: Contains:
                - 'tokens': List of tokens.
                - 'token_logprobs': List of log probabilities for each token.
        """
        if self.backend == "openai":
            return self.get_output_probabilities_openai(prompt)
        elif self.backend == "local":
            return self.get_output_probabilities_local(prompt)

    def get_output_probabilities_openai(self, prompt: str) -> Dict[str, Any]:
        """
        Retrieves token-level probabilities using the OpenAI API.
        """
        try:
            response = openai.Completion.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                logprobs=self.logprobs,
                echo=True  # Echo the prompt tokens along with the generated ones.
            )
            choice = response["choices"][0]
            tokens = choice["logprobs"]["tokens"]
            token_logprobs = choice["logprobs"]["token_logprobs"]
            return {"tokens": tokens, "token_logprobs": token_logprobs}
        except Exception as e:
            print("Error calling OpenAI API:", e)
            return {}

    def get_output_probabilities_local(self, prompt: str) -> Dict[str, Any]:
        """
        Retrieves token-level probabilities using a local Hugging Face Transformers model.
        This implementation computes the log probability of each token in the prompt.
        """
        try:
            # Tokenize the input prompt.
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            # Get model outputs without gradient tracking.
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits  # shape: [1, seq_len, vocab_size]

            # Compute log probabilities using softmax over logits.
            # Shift logits and labels to compute the probability of each token given its context.
            shift_logits = logits[:, :-1, :]  # shape: [1, seq_len-1, vocab_size]
            shift_labels = input_ids[:, 1:]     # shape: [1, seq_len-1]
            log_probs = F.log_softmax(shift_logits, dim=-1)  # shape: [1, seq_len-1, vocab_size]
            # Gather log probabilities corresponding to the actual tokens.
            token_logprobs_tensor = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # shape: [1, seq_len-1]
            token_logprobs = token_logprobs_tensor[0].tolist()
            # Convert token IDs back to tokens.
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            # The first token has no preceding context, so we insert a None.
            token_logprobs = [None] + token_logprobs
            return {"tokens": tokens, "token_logprobs": token_logprobs}
        except Exception as e:
            print("Error in local inference:", e)
            return {}

    def compute_sentence_probability(self, prompt: str, sentence: str) -> float:
    """
    Computes the probability of generating a given sentence token-by-token, conditioned on an initial prompt.

    Args:
        prompt (str): The initial context or prompt.
        sentence (str): The sentence whose probability you want to calculate.

    Returns:
        float: The computed sentence probability.
    """
    # Concatenate the prompt and sentence
    full_text = prompt + sentence

    # Tokenize prompt and full text separately
    prompt_tokens = self.tokenizer.tokenize(prompt)
    full_tokens = self.tokenizer.tokenize(full_text)

    # Sentence tokens are the tokens appearing after the prompt tokens
    sentence_tokens = full_tokens[len(prompt_tokens):]

    total_logprob = 0.0
    current_prompt = prompt  # Initially, the provided prompt

    for idx, token in enumerate(sentence_tokens):
        # Get probabilities from the current prompt
        response = self.get_output_probabilities(current_prompt)

        # Validate response
        if not response or "tokens" not in response or "token_logprobs" not in response:
            print(f"[Error] Invalid response at token index {idx} ('{token}').")
            return 0.0

        tokens = response["tokens"]
        token_logprobs = response["token_logprobs"]

        # For next-token probability, the relevant token is always the first token generated after the prompt.
        # The token we are interested in is the first token that appears after the current prompt.
        if len(tokens) <= len(self.tokenizer.tokenize(current_prompt)):
            print(f"[Error] No tokens generated beyond prompt at index {idx} ('{token}').")
            return 0.0

        # Find the position of the next predicted token
        next_token_idx = len(self.tokenizer.tokenize(current_prompt))

        generated_token = tokens[next_token_idx]
        logprob = token_logprobs[next_token_idx]

        # Check if the generated token matches the expected token
        if generated_token != token:
            print(f"[Warning] Mismatch at token index {idx}: expected '{token}', got '{generated_token}'.")

        # Accumulate log probabilities regardless of mismatch to continue the calculation
        total_logprob += logprob

        # Update prompt for next iteration
        current_prompt += token

    # Return the exponentiated total log probability
    return math.exp(total_logprob)
