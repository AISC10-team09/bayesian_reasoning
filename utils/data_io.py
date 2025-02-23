"""
This module provides functions for data input/output, logging, and saving results.
It includes:
    - setup_logging: Configures the logging system.
    - save_results: Saves a Python dictionary (e.g., experimental results) to a JSON file.
    - load_results: Loads data from a JSON file.
"""

import json
import logging
import os


def setup_logging(log_file: str = None, level: int = logging.INFO):
    """
    Set up the logging configuration.

    If log_file is provided, logs will be written to that file; otherwise, logs are output to the console.

    Args:
        log_file (str, optional): Path to a log file.
        level (int, optional): Logging level (default is logging.INFO).
    """
    handlers = []
    if log_file:
        # Ensure that the directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    else:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers
    )
    logging.info("Logging initialized. Log file: %s", log_file if log_file else "Console")


def save_results(results: dict, file_path: str):
    """
    Save the provided results dictionary to a JSON file.

    Args:
        results (dict): The data to save.
        file_path (str): The file path where the results will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)
        logging.info("Results successfully saved to '%s'.", file_path)
    except Exception as e:
        logging.error("Error saving results to '%s': %s", file_path, e)


def load_results(file_path: str) -> dict:
    """
    Load results from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The loaded data. Returns None if the file is not found or an error occurs.
    """
    if not os.path.exists(file_path):
        logging.error("File not found: '%s'.", file_path)
        return None

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        logging.info("Results successfully loaded from '%s'.", file_path)
        return data
    except Exception as e:
        logging.error("Error loading results from '%s': %s", file_path, e)
        return None


if __name__ == "__main__":
    # Example usage:
    setup_logging(log_file="logs/experiment.log")

    sample_results = {
        "experiment": "Bayesian Consistency Test",
        "prior_probabilities": {"Shakespeare": 0.6, "Mark Twain": 0.4},
        "likelihoods": {"Shakespeare": 0.7, "Mark Twain": 0.3},
        "posterior_probabilities": {"Shakespeare": 0.8, "Mark Twain": 0.2},
        "BCE": 0.05
    }

    # Save the sample results
    save_results(sample_results, "results/sample_experiment_results.json")

    # Load the sample results
    loaded_results = load_results("results/sample_experiment_results.json")
    logging.info("Loaded Results: %s", loaded_results)
