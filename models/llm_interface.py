"""
llm_interface_2.py

this module provides an interface to interact with large language models to extract token-level probability estimates. 
"""

import math
from typing import Dict, Any, Union # List
import uuid

# imports for local inference using huggingface transformers
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class LLMInterface:
    """
    a class to interface with a llm

    attributes:
        model_name (str): The model identifier (e.g., "text-davinci-003" or "gpt2").
    """

    def __init__(self, model_name: str, api_key: str = None, backend: str = "openai",
                 max_tokens: int = 50, temperature: float = 0.7, logprobs: int = 5):
        """
        initializes the LLMInterface with model settings.

        args:
            model_name (str): The model identifier.
        """
        self.model_name = model_name

        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.model.eval()  # set the model to evaluation mode.

        # Check for available accelerators
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # For Apple Silicon (M1/M2/M3)
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

    def get_output_probabilities(self, prompt: str) -> Dict[str, Any]:
        """
        retrieves token-level probability
        """
        try:
            # Move model to device if not already there
            self.model = self.model.to(self.device)
            # Encode and move to device
            tokens_input = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            # get model outputs without gradient tracking
            with torch.no_grad():
                outputs = self.model(tokens_input)
                logits = outputs.logits  # shape: [1, seq_len, vocab_size]

            # Make sure to keep tensors on the same device
            tokens_output = torch.argmax(logits, dim=-1).squeeze(0)
            # output = self.tokenizer.decode(tokens_output)
            # print("output: ", output)

            log_probs = F.log_softmax(logits, dim = -1).squeeze(0)  # shape: [1, seq_len-1, vocab_size]

            # Return tensors detached from computation graph and moved to CPU for easier handling
            return {
                "tokens": tokens_output.detach().cpu(),
                "logprobs": log_probs.detach()  # Keep on device for next computation
            }
        
        except Exception as e:
            print("error in inference:", e)
            return {}


    def compute_probability(
        self, 
        prompt: str, 
        expected_response: str, 
        log_details: bool = False, 
        experiment_id: str = None
    ) -> Union[float, Dict[str, Any]]:
        """
        computes the probability of generating a specific response token-by-token, conditioned on an initial prompt

        args:
            prompt (str): the prompt
            expected_response (str): the expected response whose probability is to be calcualted
            log_details (bool, optional): if true, returns a detailed log dictionary along with the final probability
            experiment_id (str, optional): an identifier for the experiment; if not provided, one is generated automatically.

        returns:
            if log_details is false: 
                final_probability (float): the final probability of the response
            if log_details is true:
                dict - a dictionary containing:
                  - "experiment_id": unique experiment identifier.
                  - "conversation_history": the initial prompt
                  - "data_x": the expected response
                  - "steps": list of step logs, each with:
                        * step index,
                        * current prompt,
                        * expected token,
                        * log probability,
                        * error_msg
                  - "cumulative_logprob": sum of token log probabilities.
                  - "final_probability": exponentiated cumulative log probability.
                  - "error_msg": an error message, if any
        """
    
        if experiment_id is None: experiment_id = str(uuid.uuid4())

        # tokens_prompt = self.tokenizer.tokenize(prompt)     # tokenize the prompt

        # tokenize the response
        full_text = prompt + expected_response
        full_tokens = self.tokenizer.encode(full_text, return_tensors="pt")[0].to(self.device)
        
        # Tokenize just the prompt to find where the response starts
        prompt_tokens = self.tokenizer.encode(prompt, return_tensors="pt")[0].to(self.device)
        prompt_length = len(prompt_tokens)
        
        # Get the expected response tokens
        response_tokens = full_tokens[prompt_length:]
                            
        total_logprob = 0.0
        current_prompt = prompt
        steps = []

        # print("len(tokens_expected_response):", len(tokens_expected_response))
        # print("tokens expected response:", tokens_expected_response)

        # print("len(tokens_expected_response_enc):", len(tokens_expected_response_enc))
        # print("tokens expected response end:", tokens_expected_response_enc)


        for idx, token_id in enumerate(response_tokens):

            # query the model with the current prompt        
            response = self.get_output_probabilities(current_prompt)

            # validate the response structure
            if not response or "tokens" not in response or "logprobs" not in response:
                error_msg = f"[error] invalid response at step {idx}."
                if log_details:
                    steps.append({
                        "step": idx,
                        "current_prompt": current_prompt,
                        "expected_token": self.tokenizer.decode([token_id.item()]),
                        "log_prob": 0.00,
                        "error": error_msg
                    })
                    return {
                        "experiment_id": experiment_id,
                        "conversation_history": prompt,
                        "data_x": expected_response,
                        "steps": steps,
                        "total_logprob": total_logprob,
                        "final_probability": 0.0,
                        "error": error_msg
                    }
                else:
                    print(error_msg)
                    return 0.0
            
        
            # find the log probability for the token corresponding to the expected response
            logprobs = response["logprobs"]  # shape: [seq_len, vocab_size]
            logprob = logprobs[-1, token_id].item()
            # print("logprob shape:", logprob.shape)
            token_str = self.tokenizer.decode([token_id.item()])
            # print("idx:", idx, "logprob:", logprob.item())

            # record the step details
            if log_details:
                step_log = {
                    "step": idx,
                    "current_prompt": current_prompt,
                    "expected_token": token_str,
                    "logprob": logprob,
                    "error_msg": None
                }

                steps.append(step_log)

            # determine the next prompt (to mimic autoregressive generation)
            current_prompt = current_prompt + token_str
            # prompt_current += prompt_next

            # print("prompt_next:", prompt_next)

            total_logprob += logprob    # accumulate the log probability
        final_probability = math.exp(total_logprob) if total_logprob > -float('inf') else 0.0

            # print("idx:", idx, "final probability:", final_probability)

        if log_details:
            return {
                "experiment_id": experiment_id,
                "conversation_history": prompt,
                "data_x": expected_response,
                "steps": steps,
                "total_logprob": total_logprob,
                "final_probability": final_probability,
                "error_msg": None
            }
        else:
            return final_probability
