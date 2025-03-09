"""
llm_interface.py

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
        logprobs (int): Number of log probabilities (top_logprobs) to request per token.
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
            logprobs (int, optional): Number of top_logprobs to return per token. Default is 5.
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
        Sends a prompt to the LLM and retrieves token-level probability estimates.

        For "openai", calls the OpenAI API to get token-level top_logprobs.
        For "local", uses Hugging Face Transformers to compute log probabilities.

        Args:
            prompt (str): The input prompt.

        Returns:
            dict: Contains:
                - 'tokens': List of tokens.
                - 'top_logprobs': List of log probabilities (from the top_logprobs) for each token.
        """
        if self.backend == "openai":
            return self.get_output_probabilities_openai(prompt)
        elif self.backend == "local":
            return self.get_output_probabilities_local(prompt)

    def get_output_probabilities_openai(self, prompt: str) -> Dict[str, Any]:
        """
        Retrieves token-level probability estimates using the OpenAI API.
        Expects the response to contain the 'top_logprobs' field.

        Raises:
            ValueError: If the required 'top_logprobs' field is missing.
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
            logprobs_dict = choice.get("logprobs", {})
            if "top_logprobs" not in logprobs_dict:
                earlier_tokens = logprobs_dict.get("tokens", [])
                earlier_logprobs = logprobs_dict.get("token_logprobs", [])
                cumulative_logprob = sum(lp for lp in earlier_logprobs if lp is not None)
                error_message = (
                    "Required top_logprobs not present in the response.\n"
                    f"Earlier tokens: {earlier_tokens}\n"
                    f"Cumulative log probability of earlier tokens: {cumulative_logprob}\n"
                    f"Full logprobs response: {logprobs_dict}"
                )
                raise ValueError(error_message)
            tokens = logprobs_dict["tokens"]
            top_logprobs = logprobs_dict["top_logprobs"]
            return {"tokens": tokens, "top_logprobs": top_logprobs}
        except Exception as e:
            print("Error calling OpenAI API:", e)
            return {}

    def get_output_probabilities_local(self, prompt: str) -> Dict[str, Any]:
        """
        Retrieves token-level probability estimates using a local Hugging Face Transformers model.
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
            # For consistency, we return the computed log probabilities as "top_logprobs".
            return {"tokens": tokens, "top_logprobs": token_logprobs}
        except Exception as e:
            print("Error in local inference:", e)
            return {}

    def compute_sentence_probability(
        self, 
        prompt: str, 
        sentence: str, 
        log_details: bool = False, 
        topk: int = None, 
        experiment_id: str = None
    ) -> Union[float, Dict[str, Any]]:
        """
        Computes the probability of generating a given sentence token-by-token,
        conditioned on an initial prompt. Optionally logs detailed information at each
        autoregressive step.

        Args:
            prompt (str): The initial context or prompt.
            sentence (str): The sentence whose probability is to be calculated.
            log_details (bool, optional): If True, returns a detailed log dictionary along
                                          with the final probability.
            topk (int, optional): If specified, limits the logged top_logprobs to the top k entries.
            experiment_id (str, optional): An optional identifier for the experiment; if not provided,
                                           one is generated automatically.

        Returns:
            If log_details is False: 
                float - The final probability of the sentence.
            If log_details is True:
                dict - A dictionary containing:
                  - "experiment_id": Unique experiment identifier.
                  - "conversation_history": The initial prompt.
                  - "data_X": The expected sentence.
                  - "model": Model name.
                  - "model_parameters": Dictionary of key parameters.
                  - "steps": List of step logs, each with:
                        * step index,
                        * current prompt,
                        * expected token,
                        * generated token,
                        * selected log probability,
                        * top_logprobs (optionally trimmed to topk).
                  - "cumulative_logprob": Sum of selected token log probabilities.
                  - "final_probability": Exponentiated cumulative log probability.
        """
        if experiment_id is None:
            experiment_id = str(uuid.uuid4())

        # Concatenate the prompt and sentence
        full_text = prompt + sentence

        # Tokenize prompt and full text separately
        prompt_tokens = self.tokenizer.tokenize(prompt)
        full_tokens = self.tokenizer.tokenize(full_text)

        # Sentence tokens are those appearing after the prompt tokens
        sentence_tokens = full_tokens[len(prompt_tokens):]

        total_logprob = 0.0
        current_prompt = prompt  # Start with the initial prompt
        steps = []  # To record detailed logs for each autoregressive step

        for idx, expected_token in enumerate(sentence_tokens):
            # Query the model with the current prompt
            response = self.get_output_probabilities(current_prompt)

            # Validate the response structure
            if not response or "tokens" not in response or "top_logprobs" not in response:
                error_msg = f"[Error] Invalid response at step {idx} for token '{expected_token}'."
                if log_details:
                    steps.append({
                        "step": idx,
                        "current_prompt": current_prompt,
                        "error": error_msg
                    })
                    return {
                        "experiment_id": experiment_id,
                        "conversation_history": prompt,
                        "data_X": sentence,
                        "model": self.model_name,
                        "model_parameters": {
                            "max_tokens": self.max_tokens,
                            "temperature": self.temperature,
                            "logprobs": self.logprobs
                        },
                        "steps": steps,
                        "cumulative_logprob": total_logprob,
                        "final_probability": 0.0,
                        "error": error_msg
                    }
                else:
                    print(error_msg)
                    return 0.0

            tokens = response["tokens"]
            top_logprobs = response["top_logprobs"]

            # Determine the index for the next token (i.e., after current prompt tokens)
            current_prompt_tokens = self.tokenizer.tokenize(current_prompt)
            if len(tokens) <= len(current_prompt_tokens):
                error_msg = f"[Error] No tokens generated beyond prompt at step {idx} for token '{expected_token}'."
                if log_details:
                    steps.append({
                        "step": idx,
                        "current_prompt": current_prompt,
                        "error": error_msg
                    })
                    return {
                        "experiment_id": experiment_id,
                        "conversation_history": prompt,
                        "data_X": sentence,
                        "model": self.model_name,
                        "model_parameters": {
                            "max_tokens": self.max_tokens,
                            "temperature": self.temperature,
                            "logprobs": self.logprobs
                        },
                        "steps": steps,
                        "cumulative_logprob": total_logprob,
                        "final_probability": 0.0,
                        "error": error_msg
                    }
                else:
                    print(error_msg)
                    return 0.0

            next_token_idx = len(current_prompt_tokens)
            generated_token = tokens[next_token_idx]
            selected_logprob = top_logprobs[next_token_idx]

            # Optionally trim the top_logprobs to topk entries if provided and if it's a dict/list
            if topk is not None:
                # If top_logprobs is a dict, we assume it contains more entries; otherwise, if it's a list, slice it.
                if isinstance(selected_logprob, dict):
                    trimmed_top_logprobs = dict(list(selected_logprob.items())[:topk])
                elif isinstance(selected_logprob, list):
                    trimmed_top_logprobs = selected_logprob[:topk]
                else:
                    trimmed_top_logprobs = selected_logprob
            else:
                trimmed_top_logprobs = selected_logprob

            # Log a warning if the generated token doesn't match the expected token.
            if generated_token != expected_token:
                warning_msg = f"[Warning] At step {idx}: expected '{expected_token}', got '{generated_token}'."
                print(warning_msg)
            else:
                warning_msg = None

            # Record the step details
            step_log = {
                "step": idx,
                "current_prompt": current_prompt,
                "expected_token": expected_token,
                "generated_token": generated_token,
                "selected_logprob": selected_logprob,
                "top_logprobs": trimmed_top_logprobs
            }
            if warning_msg:
                step_log["warning"] = warning_msg
            steps.append(step_log)

            # Accumulate the log probability
            total_logprob += selected_logprob

            # Update the prompt by appending the expected token (to mimic autoregressive generation)
            current_prompt += expected_token

        final_probability = math.exp(total_logprob) if total_logprob != 0 else 0.0

        if log_details:
            return {
                "experiment_id": experiment_id,
                "conversation_history": prompt,
                "data_X": sentence,
                "model": self.model_name,
                "model_parameters": {
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "logprobs": self.logprobs
                },
                "steps": steps,
                "cumulative_logprob": total_logprob,
                "final_probability": final_probability
            }
        else:
            return final_probability
