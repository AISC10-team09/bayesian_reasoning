import pandas as pd
import json
import os
import sys

# Assuming the script is run from the root of the repository,
# so 'src' should be directly importable.
from src.logprobs import collect_logprobs

# Define parameters
dataset_path = "data/test_data.json"
output_path = "data/test_logprobs.csv" # This will be the new file
models_list = ["openai-community/gpt2"]
# Note: model_kwargs and model_params are lists of dicts,
# corresponding to each model in models_list for one_to_one mapping.
model_kwargs_list = [{}] 
model_params_list = [{"temperature": 1.0, "device": "cpu"}] # Changed device to cpu as mps might not be available
model_provider_name = "hf"
param_mapping_strat = "one_to_one"

# Deleting the file is handled by the agent before running this script.
# However, a check here can be a safeguard or for direct script execution.
if os.path.exists(output_path):
    print(f"Warning: Output file {output_path} already exists. It should have been deleted before running this script.")
    # os.remove(output_path) # Optionally, uncomment to enforce deletion within the script too
    # print(f"Removed existing file: {output_path}")


# Load dataset
try:
    with open(dataset_path, "r") as f:
        dataset_data = json.load(f)
    print(f"Successfully loaded dataset from {dataset_path}")
except Exception as e:
    print(f"Error loading dataset {dataset_path}: {e}")
    dataset_data = None
    # Exit if dataset cannot be loaded, as collect_logprobs will fail
    sys.exit(1)

if dataset_data:
    # Call collect_logprobs
    print("Calling collect_logprobs...")
    try:
        df_results = collect_logprobs(
            dataset=dataset_data,
            models=models_list,
            model_kwargs=model_kwargs_list,
            model_params=model_params_list,
            model_provider=model_provider_name,
            param_mapping_strategy=param_mapping_strat,
            save_results=True, # This will call save_logprobs internally
            save_path=output_path,
            verbose=True
        )

        if df_results is not None and not df_results.empty:
            print("Generated DataFrame head:")
            print(df_results.head())
            print(f"\nSuccessfully generated {output_path}")
        else:
            print("No DataFrame was generated or it was empty.")
            sys.exit(1) # Exit with error if no dataframe is produced
            
    except Exception as e:
        print(f"Error during collect_logprobs execution: {e}")
        sys.exit(1) # Exit with error
else:
    print("Skipping collect_logprobs due to dataset loading error.")
    sys.exit(1) # Exit with error

sys.exit(0) # Ensure a clean exit if everything is successful
