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

class LLMInterface_2:
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


    def get_output_probabilities(self, prompt: str) -> Dict[str, Any]:
        """
        retrieves token-level probability
        """
        try:
 
            tokens_input = self.tokenizer.encode(prompt, return_tensors = "pt")     # tokenize and encode the input prompt

            # get model outputs without gradient tracking
            with torch.no_grad():
                outputs = self.model(tokens_input)
                logits = outputs.logits  # shape: [1, seq_len, vocab_size]

            tokens_output = torch.argmax(logits, dim = -1).squeeze(0)
            # output = self.tokenizer.decode(tokens_output)
            # print("output: ", output)

            log_probs = F.log_softmax(logits, dim = -1).squeeze(0)  # shape: [1, seq_len-1, vocab_size]

            return {"tokens": tokens_output, "logprobs": log_probs}
        
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
        tokens_expected_response = []
        expected_response_words = expected_response.split()
        # print("expected response words:", expected_response_words)
        # print(expected_response_words[0])

        for i in range(len(expected_response_words)):
            tokens = self.tokenizer.tokenize(expected_response_words[i])
            for j in range(len(tokens)): tokens_expected_response.append(tokens[j])
                            
        total_logprob = 0.0
        prompt_current = prompt  # start with the initial prompt
        steps = []  # to record detailed logs for each autoregressive step

        # print("len(tokens_expected_response):", len(tokens_expected_response))
        # print("tokens expected response:", tokens_expected_response)

        # print("len(tokens_expected_response_enc):", len(tokens_expected_response_enc))
        # print("tokens expected response end:", tokens_expected_response_enc)


        for idx, token_expected in enumerate(tokens_expected_response):

            # query the model with the current prompt        
            response = self.get_output_probabilities(prompt_current)

            # validate the response structure
            if not response or "tokens" not in response or "logprobs" not in response:
                error_msg = f"[error] invalid response at step {idx} for token '{token_expected}'."
                if log_details:
                    steps.append({
                        "step": idx,
                        "current_prompt": prompt_current,
                        "expected_token": None,
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
            
            # print("idx:", idx, "prompt current: ", prompt_current)
            # print("idx:", idx, "expected token:", token_expected)
            # print("idx:", idx, "len(tokens):", response["tokens"].shape, "len(logprobs):", response["logprobs"].shape)

            token_expected_enc = self.tokenizer.encode(token_expected)
            # print(token_expected_enc)
        
            # find the log probability for the token corresponding to the expected response
            logprobs = response["logprobs"] # shape: [response_len, vocab_size]
            logprob = logprobs[0, token_expected_enc]
            # print("logprob shape:", logprob.shape)

            # print("idx:", idx, "logprob:", logprob.item())

            # record the step details
            if log_details == True:
                step_log = {
                    "step": idx,
                    "current_prompt": prompt_current,
                    "expected_token": token_expected,
                    "logprob": logprob,
                    "error_msg": None
                }

                steps.append(step_log)

            # determine the next prompt (to mimic autoregressive generation)
            # prompt_next = tokens_expected_response[idx + 1]
            # prompt_current += prompt_next
            prompt_current = prompt + "".join(tokens_expected_response[0:idx+1])  

            # print("prompt_next:", prompt_next)

            total_logprob += logprob    # accumulate the log probability
            final_probability = math.exp(total_logprob) if total_logprob != 0 else 0.0

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
