import pandas as pd
import json
import os
import math
from typing import List, Dict, Any, Union, Tuple
from itertools import product
from tqdm import tqdm  # Add tqdm import for progress bars

# Import the specific interfaces needed
from src.models.llm_interface import LLMInterface  # Import ABC for type hinting
from src.models.hf_interface import HFInterface
from src.models.vllm_interface import VLLMInterface, VLLM_AVAILABLE

import gc
import sys
import torch

DEFAULT_LOGPROBS_FILE = "data/logprobs.csv"

def save_logprobs(logprobs: pd.DataFrame, save_path: str = DEFAULT_LOGPROBS_FILE):
    """
    Save log probabilities to a CSV file. Appends if file exists, includes header only if new/empty.
    """
    file_exists = os.path.exists(save_path)
    # Check if file exists and is non-empty to decide on header
    header = not file_exists or (
        os.path.exists(save_path) and os.path.getsize(save_path) == 0
    )
    mode = "a" if file_exists else "w"

    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    logprobs.to_csv(
        save_path, mode=mode, header=header, index=False
    )  # Don't write index


def collect_logprobs(
    dataset: Union[str, List[Dict[str, Any]]],
    models: List[str],
    model_kwargs: List[Dict[str, Any]] = None,
    model_params: List[Dict[str, Any]] = None,
    model_provider: str = "hf",
    param_mapping_strategy: str = "one_to_one",
    save_results: bool = True,
    save_path: str = DEFAULT_LOGPROBS_FILE,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute log probabilities using the specified LLM provider interface.

    Args:
        dataset (Union[str, List[Dict[str, Any]]]): Path to JSON dataset or loaded data.
        models (List[str]): List of model identifiers.
        model_kwargs (List[Dict[str, Any]], optional): List of model-specific keyword arguments
            passed directly to model loading functions (e.g., trust_remote_code, use_auth_token).
            Follows the same mapping strategy as model_params. Defaults to None (empty dict for each model).
        model_params (List[Dict[str, Any]], optional): List of parameter dicts for each model run.
            These parameters are passed directly to the chosen model interface
            (HFInterface or VLLMInterface) during initialization. Any parameters
            provided will also be included as columns in the output DataFrame.
            Common parameters include:
                - 'device' (str): Target device ('auto', 'cuda', 'cpu', 'mps'). Default: 'auto'.
            HFInterface specific parameters:
                - 'batch_size' (int): Batch size for processing. Default: 8.
                - 'temperature' (float): Temperature for scaling logits. Default: 1.0.
            VLLMInterface may accept different parameters (e.g., 'tensor_parallel_size').
        model_provider (str): The provider to use ('hf' or 'vllm'). Defaults to 'hf'.
        param_mapping_strategy (str): How to map models to parameters.
            'one_to_one': models[i] uses model_params[i]. Requires len(models) == len(model_params). (Default)
            'combinations': Each model in `models` is run with each parameter set in `model_params`.
        save_results (bool): If True, saves the resulting DataFrame.
        save_path (str): Path to save the results CSV file.
        verbose (bool): If True, print detailed progress and informational messages. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with computed log probabilities and metadata.

    Raises:
        NotImplementedError: If an unsupported model_provider is specified.
        ValueError: If 'one_to_one' strategy is used and len(models) != len(model_params).
    """
    # --- 0. Validate Inputs ---
    param_mapping_strategy = param_mapping_strategy.lower()
    if param_mapping_strategy not in ["one_to_one", "combinations"]:
        raise ValueError(
            f"Invalid param_mapping_strategy: '{param_mapping_strategy}'. Must be 'one_to_one' or 'combinations'."
        )

    # Store original None status to correctly interpret user's intent
    original_model_kwargs_is_none = model_kwargs is None
    original_model_params_is_none = model_params is None

    # Perform checks for 'one_to_one' strategy here, using effective parameters
    if param_mapping_strategy == "one_to_one":
        # For one-to-one, if params/kwargs are None, they default to list of empty dicts matching models length
        effective_params_for_check = model_params if not original_model_params_is_none else [{} for _ in range(len(models))]
        effective_kwargs_for_check = model_kwargs if not original_model_kwargs_is_none else [{} for _ in range(len(models))]

        if len(models) != len(effective_params_for_check):
            mp_info = f"originally {'None' if original_model_params_is_none else ('provided (length ' + str(len(model_params)) + ')' if isinstance(model_params, list) else 'provided (type ' + str(type(model_params)) + ')')}"
            raise ValueError(
                f"With 'one_to_one' mapping strategy, the number of models ({len(models)}) must equal the number of parameter sets ({len(effective_params_for_check)})."
                f" Check your `model_params` argument ({mp_info}), or ensure it's None to use defaults."
            )
        if len(models) != len(effective_kwargs_for_check):
            mkw_info = f"originally {'None' if original_model_kwargs_is_none else ('provided (length ' + str(len(model_kwargs)) + ')' if isinstance(model_kwargs, list) else 'provided (type ' + str(type(model_kwargs)) + ')')}"
            raise ValueError(
                f"With 'one_to_one' mapping strategy, the number of models ({len(models)}) must equal the number of model_kwargs sets ({len(effective_kwargs_for_check)})."
                f" Check your `model_kwargs` argument ({mkw_info}), or ensure it's None to use defaults."
            )
    # For "combinations" strategy, specific checks for list type if provided will be done when building the iterator.

    # --- 1. Load Dataset ---
    if isinstance(dataset, str):
        try:
            with open(dataset, "r") as f:
                loaded_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {dataset}")
            return pd.DataFrame()
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {dataset}")
            return pd.DataFrame()
    else:
        loaded_data = dataset  # Assume it's already loaded data

    # Expecting {"bayesian_reasoning": [...]} structure
    if "bayesian_reasoning" not in loaded_data:
        print("Error: 'bayesian_reasoning' key not found in dataset.")
        return pd.DataFrame()
    data_items = loaded_data["bayesian_reasoning"]

    all_results_aggregated = {}  # Use dict keyed by metadata_key to aggregate

    # Validate provider once before the loop
    model_provider = model_provider.lower()  # Convert to lowercase in-place
    if model_provider not in ["hf", "vllm"]:
        raise NotImplementedError(
            f"Model provider '{model_provider}' is not supported. Use 'hf' or 'vllm'."
        )
    if model_provider == "vllm" and not VLLM_AVAILABLE:
        raise ImportError(
            "Model provider 'vllm' requested, but vLLM library is not installed or available."
        )

    # --- Determine Iteration Strategy ---
    if param_mapping_strategy == "one_to_one":
        # Effective params already determined and validated for 'one_to_one'
        effective_model_params = model_params if not original_model_params_is_none else [{} for _ in range(len(models))]
        effective_model_kwargs = model_kwargs if not original_model_kwargs_is_none else [{} for _ in range(len(models))]
        
        model_param_iterator = list(zip(models, effective_model_params, effective_model_kwargs))
        if verbose: print("Using one-to-one model-parameter mapping.")
    else:  # "combinations"
        # For combinations, if params/kwargs are None, they default to a list containing a single empty dict
        effective_model_params = model_params if not original_model_params_is_none else [{}]
        effective_model_kwargs = model_kwargs if not original_model_kwargs_is_none else [{}]

        # Validate that if user provided model_params or model_kwargs, they are lists
        if not original_model_params_is_none and not isinstance(model_params, list):
            raise ValueError(f"If provided for 'combinations' strategy, 'model_params' must be a list of dicts. Got type: {type(model_params)}")
        if not original_model_kwargs_is_none and not isinstance(model_kwargs, list):
            raise ValueError(f"If provided for 'combinations' strategy, 'model_kwargs' must be a list of dicts. Got type: {type(model_kwargs)}")
        
        model_param_iterator = list(product(models, effective_model_params, effective_model_kwargs))
        if verbose: print("Using combinations mapping: running each model with each parameter set and each model_kwargs set.")

    # --- Sequential Processing Loop with Progress Bar ---
    iterator = tqdm(model_param_iterator, desc="Processing models", total=len(model_param_iterator))
    for model_name, params, model_kw in iterator:
        if verbose: print(f"\n--- Processing Model: {model_name} with provider: {model_provider} and params: {params} ---")
        llm: LLMInterface = None  # Type hint using the ABC
        interface_params = params.copy()  # Use the specific params for this run
        interface_params['verbose'] = verbose  # Add verbose to params passed to interface

        try:
            # --- Instantiate the correct interface based on model_provider ---
            if model_provider == "hf":
                if verbose: print(f"Using HFInterface for {model_name} with model_kwargs: {model_kw}")
                llm = HFInterface(model_name=model_name, model_kwargs=model_kw, **interface_params)
            elif model_provider == "vllm":
                if verbose: print(f"Using VLLMInterface for {model_name} with model_kwargs: {model_kw}")
                llm = VLLMInterface(model_name=model_name, model_kwargs=model_kw, **interface_params)

        except Exception as e:
            print(
                f"Error loading model {model_name} with params {params} using {model_provider.upper()}Interface: {e}. Skipping this combination."
            )
            continue  # Skip to next model/param combination

        # List to hold tuples: (type, prompt, expected, metadata, metadata_key)
        prompts_to_process: List[Tuple[str, str, str, Dict[str, Any], Tuple]] = []

        # --- 3. Iterate through Dataset Items & Generate Prompts ---
        for item_index, item in enumerate(data_items):
            item_histories = item.get("histories", [])
            item_classes = item.get("classes", [])
            raw_evidences = item.get("evidences", [])
            class_elicitation = item.get("class_elicitation", "")
            evidence_elicitation = item.get("evidence_elicitation", "")
            item_class_category = item.get("class_category", "unknown")

            # Normalize evidence into a list of dicts {evidence_text}
            normalized_evidence_list = []
            if isinstance(raw_evidences, list):
                for ev_item in raw_evidences:
                    if isinstance(ev_item, dict) and "evidence_text" in ev_item:
                        normalized_evidence_list.append(
                            {"evidence_text": ev_item["evidence_text"]}
                        )
                    # If evidence is just a list of strings (old format, less likely with new structure)
                    elif isinstance(ev_item, str):
                         normalized_evidence_list.append({"evidence_text": ev_item})

            # Skip item if no valid evidence found after normalization
            if not normalized_evidence_list:
                # if verbose: print(f"Warning: No valid evidence found for item {item_index}. Skipping.")
                continue
            
            if not item_histories: # Skip if no histories
                # if verbose: print(f"Warning: No histories found for item {item_index}. Skipping.")
                continue

            # Iterate over all combinations of history, class, and evidence
            for current_history, clas, evidence_item in product(item_histories, item_classes, normalized_evidence_list):
                evidence_text = evidence_item["evidence_text"]
                # evidence_category is removed

                # Metadata to associate results correctly
                metadata = {
                    "model_name": model_name,
                    "model_provider": model_provider,
                    "item_index": item_index,
                    "class_category": item_class_category,
                    "class": clas,
                    "evidence_text": evidence_text,
                    "conversation_history": current_history,
                    "class_elicitation": class_elicitation,
                    "evidence_elicitation": evidence_elicitation,
                    "temperature": params.get("temperature", 1.0),
                    "device": llm.device.type if hasattr(llm, "device") else params.get("device", "auto"),
                    "model_params": json.dumps(params),
                    "model_kwargs": json.dumps(model_kw),
                }
                metadata_key = tuple(sorted(metadata.items()))

                # Define prompts and full texts *once* per base key
                prior_prompt = current_history + class_elicitation
                prior_full_text = prior_prompt + clas

                likelihood_prompt = (
                    current_history + class_elicitation + clas + evidence_elicitation
                )
                likelihood_full_text = likelihood_prompt + evidence_text

                posterior_prompt = (
                    current_history + evidence_elicitation + evidence_text + class_elicitation
                )
                posterior_full_text = posterior_prompt + clas

                # Initialize result dict for this key if not seen before
                if metadata_key not in all_results_aggregated:
                    all_results_aggregated[metadata_key] = metadata.copy()
                    # Add prompts and full texts to the aggregated results
                    all_results_aggregated[metadata_key]["prior_prompt"] = prior_prompt
                    all_results_aggregated[metadata_key]["prior_full_text"] = (
                        prior_full_text
                    )
                    all_results_aggregated[metadata_key]["likelihood_prompt"] = (
                        likelihood_prompt
                    )
                    all_results_aggregated[metadata_key]["likelihood_full_text"] = (
                        likelihood_full_text
                    )
                    all_results_aggregated[metadata_key]["posterior_prompt"] = (
                        posterior_prompt
                    )
                    all_results_aggregated[metadata_key]["posterior_full_text"] = (
                        posterior_full_text
                    )

                # Prior: P(Class | History + Class Elicitation)
                # Add prompt_type to metadata_key to make it unique
                prior_key = metadata_key + (
                    ("prompt_type", "prior"),
                )  # Just append prompt type
                prompts_to_process.append(
                    ("prior", prior_prompt, clas, metadata, prior_key)
                )

                # Likelihood: P(Evidence | History + Class Elicitation + Class + Evidence Elicitation)
                likelihood_key = metadata_key + (("prompt_type", "likelihood"),)
                prompts_to_process.append(
                    (
                        "likelihood",
                        likelihood_prompt,
                        evidence_text,
                        metadata,
                        likelihood_key,
                    )
                )

                # Posterior: P(Class | History + Evidence Elicitation + Evidence + Class Elicitation)
                posterior_key = metadata_key + (("prompt_type", "posterior"),)
                prompts_to_process.append(
                    ("posterior", posterior_prompt, clas, metadata, posterior_key)
                )

        # --- 4. Process All Prompts via LLM Interface ---
        if verbose: print(f"Generated {len(prompts_to_process)} prompts for model {model_name} with params {params}. Processing via {type(llm).__name__}...")
        if not prompts_to_process:
            if verbose: print("No prompts generated for this model. Skipping.")
            if llm:
                llm.release()
            continue

        try:
            # Delegate the entire list processing to the interface
            logprob_results: Dict[
                Tuple, Union[Tuple[float, int, List[float]], None]
            ] = llm.compute_logprobs(prompts_to_process)

            # --- 5. Aggregate Results ---
            base_key_dict = {}  # Map from unique keys to base keys

            # First pass: identify base keys
            for prompt_type, _, _, metadata, unique_key in prompts_to_process:
                # The base key is everything except the last tuple element
                base_key = unique_key[:-1]
                base_key_dict[unique_key] = base_key

                # Initialize this record if not already done
                if base_key not in all_results_aggregated:
                    all_results_aggregated[base_key] = metadata.copy()

            # Second pass: populate results
            for unique_key, result_tuple in logprob_results.items():
                if unique_key in base_key_dict:
                    base_key = base_key_dict[unique_key]
                    prompt_type = unique_key[-1][1]  # Get prompt_type

                    # Handle None case and unpack tuple
                    if result_tuple is not None:
                        total_logprob, num_tokens, token_logprobs_list = result_tuple
                        all_results_aggregated[base_key][f"{prompt_type}_logprob"] = (
                            total_logprob
                        )
                        all_results_aggregated[base_key][
                            f"{prompt_type}_num_tokens"
                        ] = num_tokens
                        # Store list as string for CSV compatibility
                        all_results_aggregated[base_key][
                            f"{prompt_type}_token_logprobs"
                        ] = str(token_logprobs_list)
                    else:
                        # Mark all related fields as None if the computation failed
                        all_results_aggregated[base_key][f"{prompt_type}_logprob"] = (
                            None
                        )
                        all_results_aggregated[base_key][
                            f"{prompt_type}_num_tokens"
                        ] = None
                        all_results_aggregated[base_key][
                            f"{prompt_type}_token_logprobs"
                        ] = None

        except Exception as e:
            print(
                f"Error during batch processing via {type(llm).__name__} for model {model_name} with params {params}: {e}"
            )
            # Mark results as None if the whole call failed
            for _, _, _, _, unique_key in prompts_to_process:
                base_key = unique_key[:-1]  # Get base key
                prompt_type = unique_key[-1][1]  # Get prompt type
                if base_key in all_results_aggregated:
                    logprob_col = f"{prompt_type}_logprob"
                    num_tokens_col = f"{prompt_type}_num_tokens"  # Add num_tokens col
                    token_logprobs_col = (
                        f"{prompt_type}_token_logprobs"  # Add token_logprobs col
                    )
                    # Only mark as None if not already populated (e.g., from a previous successful batch)
                    if logprob_col not in all_results_aggregated[base_key]:
                        all_results_aggregated[base_key][logprob_col] = None
                    if num_tokens_col not in all_results_aggregated[base_key]:
                        all_results_aggregated[base_key][num_tokens_col] = None
                    if token_logprobs_col not in all_results_aggregated[base_key]:
                        all_results_aggregated[base_key][token_logprobs_col] = None

        finally:
            # --- Release Model Resources ---
            if llm:
                llm.release()
                del llm  # Ensure object is deleted
                # Optional: Force garbage collection if memory is critical
                gc.collect()
                if "torch" in sys.modules:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if (
                        hasattr(torch.backends, "mps")
                        and torch.backends.mps.is_available()
                    ):
                        torch.mps.empty_cache()

        if verbose: print(f"--- Finished processing Model: {model_name} with params: {params} ---")

    # --- 6. Create DataFrame ---
    final_results_list = list(all_results_aggregated.values())
    if not final_results_list:
        if verbose: print("No results were generated.")
        return pd.DataFrame()

    df = pd.DataFrame(final_results_list)

    # --- 7. Add Probability Columns & Reorder ---
    for p_type in ["prior", "likelihood", "posterior"]:
        logprob_col = f"{p_type}_logprob"
        prob_col = f"{p_type}_prob"
        if logprob_col in df.columns:
            # Calculate exp(logprob), handle None/NaN/errors gracefully
            df[prob_col] = df[logprob_col].apply(
                lambda x: math.exp(x)
                if pd.notna(x)
                and isinstance(x, (int, float))
                and not math.isinf(x)
                and not math.isnan(x)
                else None
            )

    # Reorder columns for clarity
    core_cols = [
        "item_index",
        "class_category",
        "class",
        "class_elicitation",
        "evidence_text",
        "evidence_elicitation",
        "conversation_history",
        "model_name",
        "model_provider",
        "temperature",
        "device",
        "model_params",
        "model_kwargs",
        "prior_logprob",
        "likelihood_logprob",
        "posterior_logprob",
        "prior_num_tokens",
        "likelihood_num_tokens",
        "posterior_num_tokens",
        "prior_prob",
        "likelihood_prob",
        "posterior_prob",
        "prior_prompt",
        "likelihood_prompt",
        "posterior_prompt",
        "prior_full_text",
        "likelihood_full_text",
        "posterior_full_text",
        "prior_token_logprobs",
        "likelihood_token_logprobs",
        "posterior_token_logprobs",
    ]
    
    # Ensure columns exist in the DataFrame before ordering
    ordered_cols = [col for col in core_cols if col in df.columns]
    # Add any other columns that might have been added unexpectedly
    ordered_cols.extend([col for col in df.columns if col not in ordered_cols])

    df = df[ordered_cols]

    # --- 8. Save Results ---
    if save_results and not df.empty:
        if verbose: print(f"Saving results to {save_path}...")
        save_logprobs(df, save_path)
    elif save_results and df.empty:
        if verbose: print("DataFrame is empty. Nothing to save.")

    return df


if __name__ == "__main__":
    collect_logprobs(
        "data/test.json",
        models=[
            "openai-community/gpt2-medium",
            "meta-llama/Llama-3.2-1B",
        ],
        model_kwargs=[
            {"trust_remote_code": False},
            {"trust_remote_code": True, "use_auth_token": True},
        ],
        model_params=[
            {"temperature": 1.0, "device": "mps", "batch_size": 4},
            {"temperature": 2.0, "device": "mps", "batch_size": 4},
        ],
        model_provider="hf",
        param_mapping_strategy="one_to_one",
        save_results=True,
        save_path="data/test_logprobs.csv",
        verbose=False,
    )
