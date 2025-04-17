import math
import time
from typing import Dict, Any, Union, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the ABC
from .llm_interface import LLMInterface

class HFInterface(LLMInterface):
    """
    Concrete implementation of LLMInterface using Hugging Face Transformers.
    Handles model loading, tokenization, length-sorted batching, and log
    probability calculation on CPU, CUDA, or MPS.
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initializes the HFInterface.

        Args:
            model_name (str): Hugging Face model identifier.
            **kwargs: Additional arguments, e.g., 'device' ('auto', 'cuda', 'cpu', 'mps'),
                      'batch_size' (int, defaults to 8).
        """
        super().__init__(model_name, **kwargs) # Call ABC init, stores kwargs

        # --- HF-specific initialization ---
        print(f"Initializing HFInterface for {self.model_name}")

        # Determine device
        requested_device = self.device_preference
        if requested_device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                 # Check if MPS is available and functional
                 try:
                     # Simple test to see if MPS works
                     torch.ones(1, device=torch.device("mps"))
                     self.device = torch.device("mps")
                 except Exception:
                     print("Warning: MPS available but may not be functional. Falling back to CPU.")
                     self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(requested_device)

        print(f"Using device: {self.device}")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"Warning: Tokenizer for {model_name} lacks a pad token. Using EOS token ({self.tokenizer.eos_token}) for padding.")
        except Exception as e:
            print(f"Error loading tokenizer for {model_name}: {e}")
            raise # Re-raise exception as tokenizer is critical

        # Load model
        try:
            # --- Add this print statement ---
            print(f"Attempting to load model with identifier: '{model_name}'")
            # ---------------------------------
            # Consider adding dtype options (e.g., torch.bfloat16 for supported GPUs)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.eval()
            self.model.to(self.device)
            print(f"Model {model_name} loaded successfully on {self.device}.")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise # Re-raise exception as model is critical


    def compute_logprobs(
        self,
        prompt_data: List[Tuple[str, str, str, Dict[str, Any], Tuple]]
        # batch_size: int = 8 # Removed: Get from self.kwargs
    ) -> Dict[Tuple, Union[Tuple[float, int, List[float]], None]]:
        """
        Computes log probabilities, token counts, and individual token logprobs
        for a list of prompts and expected responses using length-sorted batching.
        Batch size is determined from the parameters provided during initialization.

        Args:
            prompt_data (List[Tuple]): List of (type, prompt, expected, meta, meta_key).
            # batch_size: Removed.

        Returns:
            Dict[Tuple, Union[Tuple[float, int, List[float]], None]]:
                Map from metadata_key to a tuple (total_logprob, num_response_tokens, list_of_token_logprobs),
                or None on error.
        """
        results = {}
        if not prompt_data:
            return results

        # Get batch_size and temperature from initialization kwargs
        batch_size = self.kwargs.get("batch_size", 8)
        temperature = self.kwargs.get("temperature", 1.0) # Default to 1.0 (no scaling)
        print(f"Using batch size: {batch_size}, temperature: {temperature} (from init params)")

        # Validate temperature
        if not isinstance(temperature, (float, int)) or temperature <= 0:
            print(f"Warning: Invalid temperature value ({temperature}). Must be a positive number. Using default temperature 1.0.")
            temperature = 1.0

        # --- 1. Prepare and Sort by Length ---
        print(f"Preparing {len(prompt_data)} prompts for HF batching...")
        start_time = time.time()
        lengths_and_data = []
        for i, data_tuple in enumerate(prompt_data):
            prompt, expected, _, metadata_key = data_tuple[1], data_tuple[2], data_tuple[3], data_tuple[4]
            try:
                # Use combined length for sorting
                length = len(self.tokenizer.encode(prompt + expected))
                lengths_and_data.append((length, i, data_tuple))
            except Exception as e:
                print(f"Warning: Error tokenizing item for length calculation (key: {metadata_key}): {e}. Skipping.")
                results[metadata_key] = None # Mark as error

        lengths_and_data.sort(key=lambda x: x[0])
        sorted_prompt_data = [item[2] for item in lengths_and_data]
        sort_time = time.time() - start_time
        print(f"Sorting completed in {sort_time:.2f} seconds.")

        # --- 2. Process in Batches ---
        num_batches = math.ceil(len(sorted_prompt_data) / batch_size)
        print(f"Processing in {num_batches} batches of size {batch_size}...")

        for i in range(0, len(sorted_prompt_data), batch_size):
            batch_data = sorted_prompt_data[i : i + batch_size]
            if not batch_data: continue

            batch_prompts = [item[1] for item in batch_data]
            batch_expected = [item[2] for item in batch_data]
            batch_metadata_keys = [item[4] for item in batch_data]

            current_batch_results = {} # Store results for this batch temporarily
            try:
                # --- 3. Batch Tokenization ---
                full_texts = [p + e for p, e in zip(batch_prompts, batch_expected)]
                inputs = self.tokenizer(
                    full_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    return_attention_mask=True,
                ).to(self.device)

                # Get prompt lengths *after* tokenization of full texts
                # This requires re-tokenizing prompts, but ensures consistency with batch encoding
                prompt_only_tokens = self.tokenizer(batch_prompts, padding=False, truncation=True)['input_ids']
                prompt_lengths = [len(tokens) for tokens in prompt_only_tokens]

                # --- 4. Model Inference ---
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask']
                    )
                    logits = outputs.logits # Shape: (batch_size, seq_len, vocab_size)

                # --- 5. Calculate Log Probabilities ---
                # Apply temperature scaling *before* log_softmax
                # Avoid division by zero or negative temperature (already validated above)
                scaled_logits = logits / temperature

                log_probs_full = F.log_softmax(scaled_logits, dim=-1) # Use scaled logits
                input_ids = inputs['input_ids'] # Shape: (batch_size, seq_len)

                # Iterate through each item in the batch
                for batch_idx in range(len(batch_data)):
                    metadata_key = batch_metadata_keys[batch_idx]
                    prompt_len = prompt_lengths[batch_idx]
                    # Target tokens are from prompt_len onwards in the input_ids
                    # Logits predicting these tokens are from prompt_len-1 to end-1
                    start_idx = prompt_len
                    end_idx = inputs['attention_mask'][batch_idx].sum().item() # Actual sequence length without padding

                    # Ensure start_idx is not beyond the actual sequence length
                    if start_idx >= end_idx:
                         print(f"Warning: Prompt length ({prompt_len}) is >= sequence length ({end_idx}) for key {metadata_key}. Expected response might be empty or truncated. Setting logprob to 0.0.")
                         current_batch_results[metadata_key] = (0.0, 0, [])
                         continue

                    # Get the IDs of the expected response tokens within the batch context
                    response_token_ids = input_ids[batch_idx, start_idx:end_idx]
                    num_response_tokens = len(response_token_ids)

                    # Get the log probabilities for the response tokens
                    # Log prob for token at input_ids[t] is taken from log_probs_full[t-1]
                    token_logprobs = log_probs_full[batch_idx, start_idx-1:end_idx-1, :] # Shape: (resp_len, vocab_size)

                    # Gather the log probabilities of the actual response tokens
                    gathered_logprobs = torch.gather(token_logprobs, 1, response_token_ids.unsqueeze(-1)).squeeze(-1)
                    gathered_logprobs_list = gathered_logprobs.tolist()

                    # Sum the log probabilities for this response
                    total_logprob = gathered_logprobs.sum().item()

                    if math.isnan(total_logprob):
                        print(f"Warning: NaN log probability calculated for key {metadata_key}. Setting to -inf.")
                        current_batch_results[metadata_key] = (-float('inf'), num_response_tokens, gathered_logprobs_list)
                    else:
                        current_batch_results[metadata_key] = (total_logprob, num_response_tokens, gathered_logprobs_list)

            except Exception as e:
                print(f"Error processing batch starting at index {i}: {e}")
                # Mark all items in this *failed* batch as None
                for key in batch_metadata_keys:
                    if key not in current_batch_results: # Avoid overwriting results from previous successful batches
                         # EDIT: Store None for the tuple result
                         current_batch_results[key] = None
            finally:
                 # Update main results dict
                 results.update(current_batch_results)
                 # Clear cache if necessary (optional, depends on memory pressure)
                 # if self.device == torch.device("cuda"): torch.cuda.empty_cache()
                 # elif self.device == torch.device("mps"): torch.mps.empty_cache()

        print(f"Finished processing all batches for {self.model_name}.")
        return results
    

    def release(self):
        """Releases GPU memory used by the model and tokenizer."""
        print(f"Releasing HF model and tokenizer for {self.model_name} from {self.device}...")
        del self.model
        del self.tokenizer
        if self.device == torch.device("cuda"):
            torch.cuda.empty_cache()
        elif self.device == torch.device("mps"):
             # MPS cache clearing is less explicit, rely on garbage collection
             pass
        print("HF resources released.") 