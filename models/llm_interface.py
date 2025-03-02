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

    def get_output_probabilities(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Sends a prompt to the LLM and retrieves token-level probabilities.

        For "openai", calls the OpenAI API to get token-level log probabilities.
        For "local", uses Hugging Face Transformers to compute log probabilities.

        Args:
            prompt (str): The input prompt.
            response (str): The expected response from the llm.

        Returns:
            dict: Contains:
                - 'tokens': List of tokens.
                - 'token_logprobs': List of log probabilities for each token.
        """
        if self.backend == "openai":
            return self.get_output_probabilities_openai(prompt)
        elif self.backend == "local":
            # return self.get_output_probabilities_local(prompt)
            print("llm_interface() ", "prompt len: ", len(prompt), "response len: ", len(response))
            token_logprobs = self.get_output_probabilities_local(prompt, response)
            print("llm_interface() ", "num tokens returned: ", len(token_logprobs["token_logprobs"]))

            return token_logprobs

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

    def get_output_probabilities_local(self, prompt: str, resp: str) -> Dict[str, Any]:
        """
        Retrieves token-level probabilities using a local Hugging Face Transformers model.
        This implementation computes the log probability of each token in the prompt.
        """
        try:
            # Tokenize the input prompt.
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            # Tokenize the expected response
            resp_ids = self.tokenizer.encode(resp, return_tensors="pt")
            print("llm interface() ", "input_ids shape", input_ids.shape, "resp_ids shape:", resp_ids.shape)
            # Get model outputs without gradient tracking.
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits  # shape: [1, seq_len, vocab_size]
                print("llm_interface() ", "logits shape: ", logits.shape)

            # Compute log probabilities using softmax over logits.
            # Shift logits and labels to compute the probability of each token given its context.
            # shift_logits = logits[:, :-1, :]  # shape: [1, seq_len-1, vocab_size]
            shift_logits = logits[:, :resp_ids.shape[1], :] # shape: [1, resp_len, vocab_size]
            # shift_labels = input_ids[:, 1:]     # shape: [1, seq_len-1]
            shift_labels = resp_ids # shape [1, resp_len]
            print("llm_interface() ", "shift_logits shape: ", shift_logits.shape, "shift_labels shape: ", shift_labels.shape)
            log_probs = F.log_softmax(shift_logits, dim=-1)  # shape: [1, seq_len-1, vocab_size]
            # Gather log probabilities corresponding to the actual tokens.
            token_logprobs_tensor = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # shape: [1, seq_len-1]
            token_logprobs = token_logprobs_tensor[0].tolist()
            # Convert token IDs back to tokens.
            # tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            tokens = self.tokenizer.convert_ids_to_tokens(resp_ids[0])
            # The first token has no preceding context, so we insert a None.
            # token_logprobs = [None] + token_logprobs
            token_logprobs = token_logprobs
            print("llm_interface() ", "num tokens: ", len(tokens), "num logprobs: ", len(token_logprobs))
            return {"tokens": tokens, "token_logprobs": token_logprobs}
        except Exception as e:
            print("Error in local inference:", e)
            return {}

    def compute_sentence_probability(self, token_logprobs: List[float]) -> float:
        """
        Computes the overall probability of a sentence from token log probabilities.

        It sums the log probabilities (ignoring any None values) and exponentiates the sum.

        Args:
            token_logprobs (List[float]): List of token log probabilities.

        Returns:
            float: The overall sentence probability.
        """
        valid_logprobs = [lp for lp in token_logprobs if lp is not None]
        if not valid_logprobs:
            return 0.0
        total_logprob = sum(valid_logprobs)
        return math.exp(total_logprob)
