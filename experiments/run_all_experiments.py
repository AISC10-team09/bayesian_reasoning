import json
import logging
from models.llm_interface import LLMInterface

def run_experiments_from_json(json_path, model_name="gpt2", backend="local", results_dir="results/"):
    """
    Run experiments using configurations from a JSON file.
    
    Args:
        json_path (str): Path to the JSON file containing experiment configurations
        model_name (str): Name of the LLM model to use
        backend (str): Backend to use for the LLM
        results_dir (str): Directory to save results
    
    Returns:
        dict: Dictionary mapping experiment names to their results DataFrames
    """
    # Load data from JSON file
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"JSON file not found: {json_path}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in file: {json_path}")
        return None
    
    # Initialize LLM
    llm = LLMInterface(model_name=model_name, backend=backend)
    
    # Dictionary to store results
    results = {}
    
    # Iterate through all experiment configurations
    for experiment in data.get("bayesian_reasoning", []):
        class_type = experiment.get("class_type", "unknown")
        logging.info(f"Running experiment for {class_type}...")
        
        # Extract configuration
        conversation_history = experiment.get("conversation_history", "")
        candidate_classes = experiment.get("candidate_classes", [])
        
        # Handle both single evidence string and list of evidence strings
        evidence = experiment.get("evidence", "")
        evidence_list = [evidence] if isinstance(evidence, str) else evidence
        
        class_elicitation = experiment.get("class_elicitation", "")
        evidence_elicitation = experiment.get("evidence_elicitation", "")
        
        # Validate configuration
        if not candidate_classes or len(candidate_classes) < 2:
            logging.warning(f"Skipping {class_type}: Not enough candidate classes")
            continue
        
        if not evidence_list:
            logging.warning(f"Skipping {class_type}: No evidence provided")
            continue
        
        # Set up output filepath
        log_filepath = f"{results_dir}/experiment_{class_type}.csv"
        
        # Add trailing period to candidates if needed (based on your current results format)
        formatted_candidates = [
            c if c.endswith(".") else f"{c}." for c in candidate_classes
        ]
        
        # Run the experiment
        from experiments.experiment_runner import run_full_experiment_multi
        
        experiment_results_df = run_full_experiment_multi(
            conversation_history,
            formatted_candidates,
            evidence_list,
            class_elicitation,
            evidence_elicitation,
            llm,
            log_filepath=log_filepath
        )
        
        # Store results
        results[class_type] = experiment_results_df
        
        logging.info(f"Completed experiment for {class_type}")
    
    return results

if __name__ == "__main__":
    import os
    from utils.data_io import setup_logging
    
    # Set up logging
    setup_logging(log_file="logs/experiments.log")
    
    # Make sure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Run all experiments from data.json
    results = run_experiments_from_json(
        json_path="data/data.json",
        model_name="gpt2",
        backend="local",
        results_dir="results/"
    )
    
    # Print summary of results
    if results:
        logging.info("=== Summary of Experiments ===")
        for class_type, df in results.items():
            total_pairs = len(df) if df is not None else 0
            avg_bce = df['BCE'].mean() if df is not None and not df.empty else 0
            logging.info(f"{class_type}: {total_pairs} pairs, Avg BCE: {avg_bce:.4f}")
    else:
        logging.error("No results were generated")

if __name__ == "__main__":
    import os
    from utils.data_io import setup_logging
    
    # Set up logging
    setup_logging(log_file="logs/experiments.log")
    
    # Make sure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Run all experiments from data.json
    results = run_experiments_from_json(
        json_path="data/data.json",
        model_name="gpt2",
        backend="local",
        results_dir="results/"
    )
    
    # Print summary of results
    if results:
        logging.info("=== Summary of Experiments ===")
        for class_type, df in results.items():
            total_pairs = len(df) if df is not None else 0
            avg_bce = df['BCE'].mean() if df is not None and not df.empty else 0
            logging.info(f"{class_type}: {total_pairs} pairs, Avg BCE: {avg_bce:.4f}")
    else:
        logging.error("No results were generated")