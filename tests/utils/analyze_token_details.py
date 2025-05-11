"""
Script to analyze token-level details from real model inference.

This script creates a detailed analysis of how the real model tokenizes and 
assigns probabilities to the completions in our test data.
"""

import json
import os
import torch
import math
from typing import List, Dict, Any, Union, Tuple
import pandas as pd

from src.models.hf_interface import HFInterface
from src.logprobs import collect_logprobs


class DetailedLoggingInterface(HFInterface):
    """Extended interface that logs detailed token information."""
    
    def compute_logprobs(self, prompt_data):
        """Computes log probabilities with detailed token information."""
        # First, get standard results using parent method
        standard_results = super().compute_logprobs(prompt_data)
        
        # Create a directory for detailed results
        os.makedirs("results/calibration", exist_ok=True)
        
        # Add token-level detail
        detailed_results = {}
        for prompt_type, prompt, expected, metadata, meta_key in prompt_data:
            # Only process if we have results for this key
            if meta_key in standard_results:
                try:
                    # Tokenize the expected completion
                    completion_tokens = self.tokenizer.encode(expected, add_special_tokens=False)
                    token_texts = self.tokenizer.convert_ids_to_tokens(completion_tokens)
                    
                    # Get the standard results
                    total_logprob, num_tokens, token_logprobs = standard_results[meta_key]
                    
                    # Create detailed record
                    record_key = str(meta_key)  # Convert tuple to string for JSON
                    detailed_results[record_key] = {
                        'prompt_type': prompt_type,
                        'completion': expected,
                        'total_logprob': float(total_logprob),  # Ensure it's a float
                        'num_tokens': num_tokens,
                        'token_logprobs': [float(lp) for lp in token_logprobs],  # Convert to list of floats
                        'token_texts': token_texts,
                        'token_ids': completion_tokens,
                        'metadata': {
                            'class': metadata.get('class', ''),
                            'class_type': metadata.get('class_type', ''),
                            'evidence_text': metadata.get('evidence_text', '')
                        }
                    }
                except Exception as e:
                    print(f"Error processing detailed results for {meta_key}: {e}")
        
        # Save detailed results
        with open("results/calibration/detailed_token_analysis.json", 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # Return standard results to maintain compatibility
        return standard_results


def analyze_real_model_outputs():
    """Run analysis with detailed logging interface."""
    # Define paths
    test_data_path = "tests/data/small_test.json"
    results_path = "results/calibration/token_analysis_logprobs.csv"
    
    # Original HFInterface reference for patching
    original_hf_interface = HFInterface
    
    try:
        # Replace HFInterface with our detailed logging version
        import src.logprobs
        src.logprobs.HFInterface = DetailedLoggingInterface
        
        # Collect logprobs with the patched interface
        print("Running model inference with detailed token logging...")
        logprobs_df = collect_logprobs(
            test_data_path,
            models=["gpt2"],
            model_provider="hf",
            save_results=True,
            save_path=results_path,
            verbose=True
        )
        
        print(f"Collected logprobs for {len(logprobs_df)} rows")
        print(f"Detailed token analysis saved to results/calibration/detailed_token_analysis.json")
        
        # Analyze the logprobs DataFrame
        print("\nLogprob Summary:")
        print(logprobs_df[["class", "prior_logprob", "likelihood_logprob", "posterior_logprob"]].to_string())
        
        # Load and analyze detailed token data
        with open("results/calibration/detailed_token_analysis.json", 'r') as f:
            token_data = json.load(f)
        
        print(f"\nDetailed token analysis contains {len(token_data)} entries")
        
        # Group by prompt type and class for analysis
        prompt_types = {}
        for key, details in token_data.items():
            prompt_type = details['prompt_type']
            completion = details['completion']
            
            if prompt_type not in prompt_types:
                prompt_types[prompt_type] = {}
                
            prompt_types[prompt_type][completion] = {
                'token_texts': details['token_texts'],
                'token_logprobs': details['token_logprobs'],
                'total_logprob': details['total_logprob'],
                'num_tokens': details['num_tokens']
            }
        
        # Print token breakdown
        print("\nToken-level analysis:")
        for prompt_type, completions in prompt_types.items():
            print(f"\n{prompt_type.upper()} PROMPT TYPE:")
            for completion, details in completions.items():
                print(f"  Completion: {completion}")
                print(f"  Tokens ({len(details['token_texts'])}): {details['token_texts']}")
                print(f"  LogProbs: {[round(lp, 2) for lp in details['token_logprobs']]}")
                print(f"  Total: {round(details['total_logprob'], 2)}")
        
        # Generate mock template
        print("\nGenerated Mock Template:")
        print("def compute_logprobs_mock(prompt_data):")
        print("    results = {}")
        print("    for prompt_type, prompt, expected, metadata, meta_key in prompt_data:")
        print("        # Generate a realistic result tuple based on prompt type and expected text")
        print("        if prompt_type == \"prior\":")
        for completion, details in prompt_types.get('prior', {}).items():
            class_name = completion.strip().replace('.', '').replace('-', '_').replace(' ', '_')
            print(f"            if \"{class_name.split('_')[0]}\" in expected:")
            print(f"                results[meta_key] = ({round(details['total_logprob'], 1)}, {details['num_tokens']}, {[round(lp, 1) for lp in details['token_logprobs']]})")
        
        print("        elif prompt_type == \"likelihood\":")
        for completion, details in prompt_types.get('likelihood', {}).items():
            print(f"                results[meta_key] = ({round(details['total_logprob'], 1)}, {details['num_tokens']}, {[round(lp, 1) for lp in details['token_logprobs']]})")
            break  # Just one example for likelihood
        
        print("        elif prompt_type == \"posterior\":")
        for completion, details in prompt_types.get('posterior', {}).items():
            class_name = completion.strip().replace('.', '').replace('-', '_').replace(' ', '_')
            print(f"            if \"{class_name.split('_')[0]}\" in expected:")
            print(f"                results[meta_key] = ({round(details['total_logprob'], 1)}, {details['num_tokens']}, {[round(lp, 1) for lp in details['token_logprobs']]})")
            
        print("    return results")
        
    finally:
        # Restore the original HFInterface
        src.logprobs.HFInterface = original_hf_interface


if __name__ == "__main__":
    analyze_real_model_outputs()