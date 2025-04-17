import math
import torch
from typing import Dict, Any, Union, List, Tuple

# Attempt to import vLLM, but don't fail if it's not installed
try:
    import vllm
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    print("Warning: vLLM library not found. VLLMInterface will not be functional.")
    LLM = None # Define dummy classes/vars if import fails
    SamplingParams = None
    VLLM_AVAILABLE = False

# Import the ABC
from .llm_interface import LLMInterface


class VLLMInterface(LLMInterface):
    """
    Concrete implementation of LLMInterface using the vLLM engine.
    Leverages vLLM's continuous batching for efficient inference.
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initializes the VLLMInterface.

        Args:
            model_name (str): Hugging Face model identifier compatible with vLLM.
            **kwargs: Additional arguments passed to vllm.LLM engine
                      (e.g., tensor_parallel_size, dtype, gpu_memory_utilization).
                      See vLLM documentation for options.
        """
        super().__init__(model_name, **kwargs) # Call ABC init

        if not VLLM_AVAILABLE:
            raise ImportError("vLLM library is required for VLLMInterface but not installed.")

        # --- vLLM-specific initialization ---
        print(f"Initializing VLLMInterface for {self.model_name}")
        # Pass relevant kwargs directly to vLLM engine
        # Filter out kwargs meant only for the ABC if necessary
        vllm_engine_kwargs = {k: v for k, v in kwargs.items() if k != 'device'} # Example filter

        try:
            # Consider adding more engine arguments from kwargs as needed
            self.llm_engine = LLM(
                model=self.model_name,
                **vllm_engine_kwargs
                # Example args:
                # tensor_parallel_size=1,
                # dtype="auto",
                # gpu_memory_utilization=0.90
            )
            # vLLM handles tokenizer internally
            print(f"vLLM engine for {model_name} initialized successfully.")
        except Exception as e:
            print(f"Error initializing vLLM engine for {model_name}: {e}")
            raise

    def compute_logprobs(
        self,
        prompt_data: List[Tuple[str, str, str, Dict[str, Any], Tuple]],
        batch_size: int # Hint ignored by vLLM's continuous batching
    ) -> Dict[Tuple, Union[float, None]]:
        """
        Computes log probabilities using the vLLM engine.

        Args:
            prompt_data (List[Tuple]): List of (type, prompt, expected, meta, meta_key).
            batch_size (int): Ignored by vLLM.

        Returns:
            Dict[Tuple, Union[float, None]]: Map from metadata_key to log probability, or None on error.
        """
        results = {}
        if not prompt_data:
            return results

        print(f"Preparing {len(prompt_data)} prompts for vLLM engine...")

        prompts_for_vllm = []
        metadata_keys_ordered = [] # Keep track of order for result mapping
        prompt_lengths = []       # Store prompt token lengths
        response_lengths = []     # Store response token lengths

        # --- 1. Prepare vLLM Requests ---
        # We need prompt logprobs. vLLM returns logprobs for the *prompt* tokens.
        # To get logprobs for the response, we feed the *full text* (prompt + response)
        # as the prompt to vLLM and extract the relevant part of prompt_logprobs.
        tokenizer = self.llm_engine.get_tokenizer() # Get tokenizer from engine
        for _, prompt_text, expected_response, _, metadata_key in prompt_data:
            full_text = prompt_text + expected_response
            prompts_for_vllm.append(full_text)
            metadata_keys_ordered.append(metadata_key)
            # Pre-calculate token lengths needed for slicing results
            try:
                 prompt_tokens = tokenizer.encode(prompt_text)
                 full_tokens = tokenizer.encode(full_text)
                 prompt_lengths.append(len(prompt_tokens))
                 # Response length is difference, ensure non-negative
                 response_lengths.append(max(0, len(full_tokens) - len(prompt_tokens)))
            except Exception as e:
                 print(f"Warning: Error tokenizing for length pre-calculation (key: {metadata_key}): {e}. Skipping.")
                 # Mark lengths that cause errors to skip later
                 prompt_lengths.append(-1)
                 response_lengths.append(-1)
                 results[metadata_key] = None # Mark as error early


        # --- 2. Set Sampling Parameters ---
        # We need logprobs, not generation. Set max_tokens=0.
        # Request logprobs for the prompt tokens. The number needed depends on sequence length.
        # Setting prompt_logprobs >= 1 should be sufficient if vLLM returns all. Check vLLM docs.
        sampling_params = SamplingParams(
            max_tokens=0,          # Don't generate anything
            logprobs=1,            # Request at least the top logprob for each prompt token
            prompt_logprobs=1      # Request logprobs for the prompt tokens (our full_text)
        )
        print("Submitting requests to vLLM engine...")

        # --- 3. Run vLLM Inference ---
        try:
            request_outputs = self.llm_engine.generate(prompts_for_vllm, sampling_params, use_tqdm=True)
        except Exception as e:
            print(f"Error during vLLM generation: {e}")
            # Mark all requests as failed if the whole call fails
            for key in metadata_keys_ordered:
                 if key not in results: # Avoid overwriting previous errors
                     results[key] = None
            return results

        print("Processing vLLM results...")
        # --- 4. Process Results ---
        if len(request_outputs) != len(metadata_keys_ordered):
             print(f"Error: Mismatch between number of vLLM outputs ({len(request_outputs)}) and requests ({len(metadata_keys_ordered)}).")
             # Mark all as error in case of mismatch
             for key in metadata_keys_ordered: results[key] = None
             return results


        for i, output in enumerate(request_outputs):
            metadata_key = metadata_keys_ordered[i]
            prompt_len = prompt_lengths[i]
            resp_len = response_lengths[i]

            # Skip if marked as error during tokenization or if output indicates error
            if metadata_key in results and results[metadata_key] is None:
                continue
            if output is None or output.prompt_logprobs is None:
                print(f"Warning: Missing output or prompt_logprobs for key {metadata_key}. Skipping.")
                results[metadata_key] = None
                continue
            if resp_len == 0: # Handle empty expected response
                 results[metadata_key] = 0.0
                 continue


            prompt_logprobs = output.prompt_logprobs # List of dicts: [{token_id: logprob}, ...]

            # Ensure we have enough logprobs returned
            # Logprobs correspond to predicting token i+1 from token i.
            # We need logprobs from index prompt_len-1 up to prompt_len + resp_len - 2
            required_logprobs_len = prompt_len + resp_len -1
            if len(prompt_logprobs) < required_logprobs_len:
                 print(f"Warning: Insufficient logprobs returned ({len(prompt_logprobs)} < {required_logprobs_len}) for key {metadata_key}. Skipping.")
                 results[metadata_key] = None
                 continue


            total_logprob = 0.0
            try:
                # Iterate through the indices corresponding to the response tokens
                for resp_idx in range(resp_len):
                    # The logprob for the resp_idx-th token of the response is at
                    # index (prompt_len + resp_idx - 1) in prompt_logprobs.
                    logprob_idx = prompt_len + resp_idx - 1

                    # Get the dictionary of logprobs for this position
                    logprob_dict_at_pos = prompt_logprobs[logprob_idx]

                    # Get the actual token ID that *was* the input at the *next* position
                    # This is the token whose probability we need.
                    target_token_id = output.prompt_token_ids[logprob_idx + 1]

                    # Find the logprob of the target_token_id at this position
                    logprob_for_token = logprob_dict_at_pos.get(target_token_id)

                    if logprob_for_token is None:
                        # This case might happen if vLLM doesn't return the logprob for the actual next token
                        # (e.g., if only top-k logprobs are returned and the actual token wasn't in the top-k).
                        # Check vLLM documentation on prompt_logprobs content. Assuming it contains the needed one.
                        print(f"Warning: Logprob for target token {target_token_id} not found at position {logprob_idx} for key {metadata_key}. Setting to -inf for this token.")
                        # If this happens often, vLLM's logprob mechanism might not be suitable directly.
                        total_logprob = -float('inf') # Mark sequence as problematic
                        break # Stop summing for this sequence

                    total_logprob += logprob_for_token

                # Check for NaN/inf after summing
                if math.isnan(total_logprob) or math.isinf(total_logprob):
                     print(f"Warning: Final logprob is NaN or Inf for key {metadata_key}. Setting to -inf.")
                     results[metadata_key] = -float('inf')
                else:
                     results[metadata_key] = total_logprob

            except IndexError as e:
                 print(f"Error processing logprobs (IndexError: {e}) for key {metadata_key} at index {i}. Skipping.")
                 results[metadata_key] = None
            except Exception as e:
                 print(f"Unexpected error processing logprobs ({type(e).__name__}: {e}) for key {metadata_key} at index {i}. Skipping.")
                 results[metadata_key] = None


        print(f"Finished processing vLLM results for {self.model_name}.")
        return results

    def release(self):
        """Releases the vLLM engine (if possible/needed)."""
        # vLLM engine management might be different; consult vLLM docs.
        # Explicit deletion might help release GPU memory sooner.
        print(f"Releasing vLLM engine for {self.model_name}...")
        del self.llm_engine
        if torch.cuda.is_available(): # vLLM primarily uses CUDA
            torch.cuda.empty_cache()
        print("vLLM resources released.") 