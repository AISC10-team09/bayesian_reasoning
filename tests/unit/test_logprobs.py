"""
Unit tests for the logprobs module in src/logprobs.py
"""

import pytest
import pandas as pd
import os
import json
from unittest.mock import patch, MagicMock
from src.logprobs import collect_logprobs, save_logprobs


class TestSaveLogprobs:
    """Tests for the save_logprobs function."""
    
    def test_new_file_creation(self, tmp_csv_path):
        """Test creating a new logprobs file."""
        # Create a sample DataFrame
        df = pd.DataFrame({
            "model_name": ["test-model"],
            "prior_logprob": [-5.0],
            "likelihood_logprob": [-3.0],
            "posterior_logprob": [-4.0]
        })
        
        # Save to a new file
        save_logprobs(df, tmp_csv_path)
        
        # Check that the file was created
        assert os.path.exists(tmp_csv_path)
        
        # Check that the content is correct
        saved_df = pd.read_csv(tmp_csv_path)
        assert len(saved_df) == 1
        assert "model_name" in saved_df.columns
        assert saved_df.iloc[0]["prior_logprob"] == -5.0
        
    def test_appending_to_existing_file(self, tmp_csv_path):
        """Test appending to an existing logprobs file."""
        # Create an initial DataFrame
        initial_df = pd.DataFrame({
            "model_name": ["model-1"],
            "prior_logprob": [-5.0],
            "likelihood_logprob": [-3.0],
            "posterior_logprob": [-4.0]
        })
        
        # Save to a new file
        save_logprobs(initial_df, tmp_csv_path)
        
        # Create a second DataFrame to append
        append_df = pd.DataFrame({
            "model_name": ["model-2"],
            "prior_logprob": [-6.0],
            "likelihood_logprob": [-2.0],
            "posterior_logprob": [-3.0]
        })
        
        # Append to the existing file
        save_logprobs(append_df, tmp_csv_path)
        
        # Check that the content includes both entries
        saved_df = pd.read_csv(tmp_csv_path)
        assert len(saved_df) == 2
        assert set(saved_df["model_name"].tolist()) == {"model-1", "model-2"}


class TestCollectLogprobs:
    """Tests for the collect_logprobs function."""
    
    @pytest.fixture
    def mock_llm_interface(self):
        """Create a mock LLM interface for testing."""
        mock_interface = MagicMock()

        # Create a dynamic return_value function that generates keys based on input
        def compute_logprobs_mock(prompt_data):
            results = {}
            for prompt_type, prompt, expected, metadata, meta_key in prompt_data:
                # Generate a realistic result tuple: (logprob, num_tokens, token_logprobs)
                if prompt_type == "prior":
                    results[meta_key] = (-5.0, 3, [-1.5, -1.8, -1.7])
                elif prompt_type == "likelihood":
                    results[meta_key] = (-3.0, 8, [-0.4, -0.3, -0.5, -0.2, -0.4, -0.6, -0.3, -0.3])
                elif prompt_type == "posterior":
                    results[meta_key] = (-4.0, 3, [-1.2, -1.3, -1.5])
            return results

        # Set the mock to use our dynamic function
        mock_interface.compute_logprobs.side_effect = compute_logprobs_mock
        mock_interface.device = type('obj', (object,), {'type': 'cpu'})
        return mock_interface
    
    @patch("src.logprobs.HFInterface")
    def test_collect_logprobs_with_mock_interface(self, mock_hf_class, mock_llm_interface, test_data_json, tmp_path):
        """Test collecting logprobs with a mock interface."""
        # Create a temp file path
        tmp_csv_path = os.path.join(tmp_path, "test_output.csv")

        # Set up the mock
        mock_hf_class.return_value = mock_llm_interface

        # Run collect_logprobs with mock
        result_df = collect_logprobs(
            test_data_json,
            models=["test-model"],
            model_provider="hf",
            save_results=True,
            save_path=tmp_csv_path,
            verbose=False
        )

        # Check that the result is not empty
        assert not result_df.empty

        # Check that compute_logprobs was called
        assert mock_llm_interface.compute_logprobs.called

        # Check that the file was saved
        assert os.path.exists(tmp_csv_path)

        # Verify some of the data in the result
        assert "prior_logprob" in result_df.columns
        assert "likelihood_logprob" in result_df.columns
        assert "posterior_logprob" in result_df.columns
        
    def test_collect_logprobs_invalid_dataset(self, tmp_csv_path):
        """Test collecting logprobs with an invalid dataset path."""
        # Try with a non-existent file
        result_df = collect_logprobs(
            "/non/existent/path.json",
            models=["test-model"],
            save_results=True,
            save_path=tmp_csv_path,
            verbose=False
        )
        
        # Should return an empty DataFrame
        assert result_df.empty
        
    def test_collect_logprobs_invalid_json(self, tmp_path, tmp_csv_path):
        """Test collecting logprobs with invalid JSON."""
        # Create an invalid JSON file
        invalid_json_path = os.path.join(tmp_path, "invalid.json")
        with open(invalid_json_path, "w") as f:
            f.write("Not valid JSON")
        
        # Try with the invalid file
        result_df = collect_logprobs(
            invalid_json_path,
            models=["test-model"],
            save_results=True,
            save_path=tmp_csv_path,
            verbose=False
        )
        
        # Should return an empty DataFrame
        assert result_df.empty
        
    def test_collect_logprobs_missing_bayesian_key(self, tmp_path, tmp_csv_path):
        """Test collecting logprobs with JSON missing the bayesian_reasoning key."""
        # Create a JSON file without bayesian_reasoning key
        invalid_data_path = os.path.join(tmp_path, "invalid_data.json")
        with open(invalid_data_path, "w") as f:
            json.dump({"some_other_key": []}, f)
        
        # Try with the invalid data structure
        result_df = collect_logprobs(
            invalid_data_path,
            models=["test-model"],
            save_results=True,
            save_path=tmp_csv_path,
            verbose=False
        )
        
        # Should return an empty DataFrame
        assert result_df.empty
    
    @patch("src.logprobs.HFInterface")
    def test_collect_logprobs_one_to_one_mapping(self, mock_hf_class, mock_llm_interface, test_data_json, tmp_csv_path):
        """Test one-to-one parameter mapping."""
        # Set up the mock
        mock_hf_class.return_value = mock_llm_interface
        
        # Run collect_logprobs with one-to-one parameter mapping
        collect_logprobs(
            test_data_json,
            models=["model-1", "model-2"],
            model_params=[{"param1": "value1"}, {"param2": "value2"}],
            param_mapping_strategy="one_to_one",
            save_results=True,
            save_path=tmp_csv_path,
            verbose=False
        )
        
        # Check that HFInterface was called with the correct parameters
        assert mock_hf_class.call_count == 2
        mock_hf_class.assert_any_call(model_name="model-1", model_kwargs={}, param1="value1", verbose=False)
        mock_hf_class.assert_any_call(model_name="model-2", model_kwargs={}, param2="value2", verbose=False)
        
    @patch("src.logprobs.HFInterface")
    def test_collect_logprobs_combinations_mapping(self, mock_hf_class, mock_llm_interface, test_data_json, tmp_path):
        """Test combinations parameter mapping."""
        # Create a temp file path
        tmp_csv_path = os.path.join(tmp_path, "test_output.csv")

        # Set up the mock
        mock_hf_class.return_value = mock_llm_interface

        # Run collect_logprobs with combinations parameter mapping
        collect_logprobs(
            test_data_json,
            models=["model-1", "model-2"],
            model_params=[{"param1": "value1"}, {"param2": "value2"}],
            param_mapping_strategy="combinations",
            save_results=True,
            save_path=tmp_csv_path,
            verbose=False
        )

        # The actual call count is 8 (2 models * 2 param sets * 2 items in test_data)
        assert mock_hf_class.call_count == 8
        # Verify the combinations of models and parameters
        # Each combination is called exactly once per test data item (total 2 items)
        calls = mock_hf_class.call_args_list
        model1_param1_count = sum(1 for call in calls if call[1]['model_name'] == 'model-1' and 'param1' in call[1])
        model1_param2_count = sum(1 for call in calls if call[1]['model_name'] == 'model-1' and 'param2' in call[1])
        model2_param1_count = sum(1 for call in calls if call[1]['model_name'] == 'model-2' and 'param1' in call[1])
        model2_param2_count = sum(1 for call in calls if call[1]['model_name'] == 'model-2' and 'param2' in call[1])

        # Each combination should be called twice (once per test data item)
        assert model1_param1_count == 2
        assert model1_param2_count == 2
        assert model2_param1_count == 2
        assert model2_param2_count == 2

        # Verify specific call parameters
        mock_hf_class.assert_any_call(model_name="model-1", model_kwargs={}, param1="value1", verbose=False)
        mock_hf_class.assert_any_call(model_name="model-1", model_kwargs={}, param2="value2", verbose=False)
        mock_hf_class.assert_any_call(model_name="model-2", model_kwargs={}, param1="value1", verbose=False)
        mock_hf_class.assert_any_call(model_name="model-2", model_kwargs={}, param2="value2", verbose=False)
    
    @patch("src.logprobs.HFInterface")
    def test_collect_logprobs_with_model_kwargs(self, mock_hf_class, mock_llm_interface, test_data_json, tmp_csv_path):
        """Test collecting logprobs with model_kwargs."""
        # Set up the mock
        mock_hf_class.return_value = mock_llm_interface
        
        # Run collect_logprobs with model_kwargs
        collect_logprobs(
            test_data_json,
            models=["test-model"],
            model_kwargs=[{"trust_remote_code": True}],
            save_results=True,
            save_path=tmp_csv_path,
            verbose=False
        )
        
        # Check that HFInterface was called with the correct model_kwargs
        mock_hf_class.assert_called_with(
            model_name="test-model", 
            model_kwargs={"trust_remote_code": True}, 
            verbose=False
        )
        
    @patch("src.logprobs.VLLM_AVAILABLE", True)
    @patch("src.logprobs.VLLMInterface")
    def test_collect_logprobs_with_vllm(self, mock_vllm_class, mock_llm_interface, test_data_json, tmp_csv_path):
        """Test collecting logprobs using vLLM."""
        # Set up the mock
        mock_vllm_class.return_value = mock_llm_interface
        
        # Run collect_logprobs with vLLM
        collect_logprobs(
            test_data_json,
            models=["test-model"],
            model_provider="vllm",
            save_results=True,
            save_path=tmp_csv_path,
            verbose=False
        )
        
        # Check that VLLMInterface was called
        assert mock_vllm_class.called
    
    def test_collect_logprobs_invalid_model_provider(self, test_data_json, tmp_csv_path):
        """Test collecting logprobs with an invalid model provider."""
        # Try with an invalid model provider
        with pytest.raises(NotImplementedError):
            collect_logprobs(
                test_data_json,
                models=["test-model"],
                model_provider="invalid",
                save_results=True,
                save_path=tmp_csv_path,
                verbose=False
            )
    
    @patch("src.logprobs.VLLM_AVAILABLE", False)
    def test_collect_logprobs_vllm_not_available(self, test_data_json, tmp_csv_path):
        """Test collecting logprobs with vLLM when it's not available."""
        # Try with vLLM when it's not available
        with pytest.raises(ImportError):
            collect_logprobs(
                test_data_json,
                models=["test-model"],
                model_provider="vllm",
                save_results=True,
                save_path=tmp_csv_path,
                verbose=False
            )