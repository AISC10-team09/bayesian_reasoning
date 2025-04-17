from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Union

class LLMInterface(ABC):
    """
    Abstract Base Class for Large Language Model interfaces.
    Defines the common contract for interacting with different LLM backends
    for log probability calculations.
    """

    @abstractmethod
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the interface, load the model and tokenizer specific
        to the backend implementation.

        Args:
            model_name (str): Identifier for the model to load.
            **kwargs: Backend-specific parameters (e.g., device, engine_config, verbose).
        """
        self.model_name = model_name
        # Store verbose flag, default to True if not provided
        self.verbose = kwargs.get("verbose", False)
        if self.verbose: print(f"Initializing LLMInterface for model: {model_name} with params: {kwargs}")
        # Store all kwargs for potential use by subclasses
        self.kwargs = kwargs
        # Example: Store common preference, subclasses can override or use kwargs directly
        self.device_preference = kwargs.get("device", "auto")

    @abstractmethod
    def compute_logprobs(
        self,
        prompt_data: List[Tuple[str, str, str, Dict[str, Any], Tuple]]
    ) -> Dict[Tuple, Union[Tuple[float, int, List[float]], None]]:
        """
        Computes log probabilities, token counts, and individual token logprobs
        for a list of prompts and expected responses.
        Each concrete implementation will handle batching and inference according
        to the specifics of its backend and initialization parameters.

        Args:
            prompt_data (List[Tuple]): List of (type, prompt, expected, meta, meta_key).

        Returns:
            Dict[Tuple, Union[Tuple[float, int, List[float]], None]]:
                Map from metadata_key to a tuple (total_logprob, num_response_tokens, list_of_token_logprobs),
                or None if calculation failed for an item.
        """
        pass

    # Optional: Define common utility methods or properties if applicable
    # For example, a method to explicitly release resources
    def release(self):
        """Optional method to release resources like GPU memory."""
        if self.verbose: # Make conditional
            print(f"Releasing resources for {self.model_name} (if applicable).")
        pass

    # You might retain compute_probability/sentence_probability here if you want
    # a default implementation that calls compute_logprobs_for_list,
    # or make them abstract as well if implementations differ significantly.
    # For simplicity, we'll keep them out of the ABC for now and implement
    # them in the concrete classes where needed.
