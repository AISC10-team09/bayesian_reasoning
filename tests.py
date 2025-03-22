"""
Tests for the Bayesian reasoning project codebase.
Tests include:
  - LLM interface (using a dummy implementation).
  - Prompt templates.
  - BCE calculations.
  - KL divergence calculations.
  - Statistical and scaling analyses (descriptive stats, regression, confidence intervals, ANOVA, correlation).
  - Experiment runner functionality.
  - Prompt variations management.
  - Data I/O (saving and loading JSON).
  - Generic plotting utilities (ensuring no exceptions are raised).

Run these tests with:
    python tests.py
"""

import os
import math
import json
import unittest
import tempfile
import numpy as np
import matplotlib.pyplot as plt

# Import modules from the project.
from models.llm_interface import LLMInterface
from models.prompt_templates import generate_prior_prompt, generate_likelihood_prompt, generate_posterior_prompt
from analysis.bce_calculations import compute_log_ratio, compute_bce
from analysis.kl_divergence import compute_kl_divergence
from analysis.scaling_analysis import (
    compute_descriptive_stats,
    compute_confidence_interval,
    bootstrap_confidence_interval,
    one_sample_ttest,
    perform_regression_analysis,
    fit_power_law,
    correlation_analysis,
    anova_test,
    plot_bce_vs_model_size,
    fit_power_law_scaling,
    plot_power_law_fit,
    plot_bce_distribution
)
from experiments.experiment_runner import run_full_experiment
from experiments.prompt_variations import get_all_variations, generate_all_prompt_variations, get_random_prompt_variation
from utils.data_io import save_results, load_results, setup_logging
from utils.plot import plot_histogram, plot_scatter, plot_line_chart

import json
from jsonschema import validate
from jsonschema.exceptions import ValidationError, SchemaError


# Create a dummy LLM interface that always returns fixed token log probabilities.
class DummyLLMInterface(LLMInterface):
    """A dummy LLMInterface for testing purposes that returns fixed responses."""

    def __init__(self, fixed_value=0.1, **kwargs):
        # We use the local backend (though we won't actually call a model)
        super().__init__(model_name="gpt2", backend="local", **kwargs)
        self.fixed_value = fixed_value

    def get_output_probabilities(self, prompt: str):
        # Simulate a response with 5 tokens, each having log probability log(fixed_value).
        token_logprobs = [math.log(self.fixed_value)] * 5
        tokens = ["token1", "token2", "token3", "token4", "token5"]
        return {"tokens": tokens, "token_logprobs": token_logprobs}


class TestLLMInterface(unittest.TestCase):
    def test_compute_sentence_probability(self):
        dummy_llm = DummyLLMInterface(fixed_value=0.1)
        # 5 tokens each with probability 0.1 => overall probability = 0.1^5.
        token_logprobs = [math.log(0.1)] * 5
        expected = math.exp(5 * math.log(0.1))
        result = dummy_llm.compute_sentence_probability(token_logprobs)
        self.assertAlmostEqual(result, expected, places=6)

    def test_get_output_probabilities(self):
        dummy_llm = DummyLLMInterface(fixed_value=0.2)
        response = dummy_llm.get_output_probabilities("Test prompt")
        self.assertIn("tokens", response)
        self.assertIn("token_logprobs", response)
        self.assertEqual(len(response["tokens"]), 5)
        self.assertEqual(len(response["token_logprobs"]), 5)


class TestPromptTemplates(unittest.TestCase):
    def test_generate_prior_prompt(self):
        history = "History text."
        candidate1 = "Candidate1"
        candidate2 = "Candidate2"
        prompt = generate_prior_prompt(history, candidate1, candidate2)
        self.assertIn(history, prompt)
        self.assertIn(candidate1, prompt)
        self.assertIn(candidate2, prompt)

    def test_generate_likelihood_prompt(self):
        history = "History text."
        candidate = "Candidate"
        evidence = "Evidence text."
        prompt = generate_likelihood_prompt(history, candidate, evidence)
        self.assertIn(history, prompt)
        self.assertIn(candidate, prompt)
        self.assertIn(evidence, prompt)

    def test_generate_posterior_prompt(self):
        history = "History text."
        candidate1 = "Candidate1"
        candidate2 = "Candidate2"
        evidence = "Evidence text."
        prompt = generate_posterior_prompt(history, evidence, candidate1, candidate2)
        self.assertIn(history, prompt)
        self.assertIn(evidence, prompt)
        self.assertIn(candidate1, prompt)
        self.assertIn(candidate2, prompt)


class TestBCECalculations(unittest.TestCase):
    def test_compute_log_ratio(self):
        result = compute_log_ratio(0.5, 0.25)
        self.assertAlmostEqual(result, math.log(2), places=4)

    def test_compute_bce(self):
        # Use probabilities such that Bayesian consistency holds.
        prior_c1, prior_c2 = 0.6, 0.4
        likelihood_c1, likelihood_c2 = 0.7, 0.3
        posterior_c1, posterior_c2 = 0.8, 0.2
        result = compute_bce(prior_c1, prior_c2, likelihood_c1, likelihood_c2, posterior_c1, posterior_c2)
        self.assertAlmostEqual(result, 0.0, places=6)


class TestKLDivergence(unittest.TestCase):
    def test_compute_kl_divergence(self):
        p = [0.7, 0.2, 0.1]
        q = [0.6, 0.25, 0.15]
        kl = compute_kl_divergence(p, q)
        expected_kl = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))
        self.assertAlmostEqual(kl, expected_kl, places=6)


class TestScalingAnalysis(unittest.TestCase):
    def test_descriptive_stats(self):
        data = [1, 2, 3, 4, 5]
        stats_dict = compute_descriptive_stats(data)
        self.assertEqual(stats_dict['n'], 5)
        self.assertAlmostEqual(stats_dict['mean'], 3.0, places=6)

    def test_confidence_interval(self):
        data = [1, 2, 3, 4, 5]
        ci = compute_confidence_interval(data)
        self.assertTrue(ci[0] < ci[1])

    def test_bootstrap_confidence_interval(self):
        data = [1, 2, 3, 4, 5]
        boot_ci = bootstrap_confidence_interval(data, num_samples=500, random_seed=123)
        self.assertTrue(boot_ci[0] < boot_ci[1])

    def test_one_sample_ttest(self):
        data = [1, 1, 1, 1, 1]
        t_stat, p_val = one_sample_ttest(data, hypothesized_mean=1)
        self.assertAlmostEqual(t_stat, 0.0, places=6)

    def test_linear_regression(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        reg_result = perform_regression_analysis(x, y)
        self.assertAlmostEqual(reg_result.slope, 2.0, places=6)

    def test_fit_power_law(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 8, 18, 32, 50]  # Approximately 2 * x^2.
        a, b, _ = fit_power_law(x, y)
        self.assertAlmostEqual(b, 2.0, places=1)

    def test_correlation_analysis(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        corr_coef, _ = correlation_analysis(x, y)
        self.assertAlmostEqual(corr_coef, 1.0, places=6)

    def test_anova(self):
        group1 = [1, 2, 3]
        group2 = [2, 3, 4]
        group3 = [3, 4, 5]
        f_stat, p_val = anova_test(group1, group2, group3)
        self.assertTrue(p_val > 0)  # p-value should be computed.

    def test_plot_functions(self):
        # Ensure plot functions run without error.
        model_sizes = [100, 200, 300, 400, 500]
        bce_values = [0.2, 0.15, 0.12, 0.10, 0.08]
        try:
            plot_bce_vs_model_size(model_sizes, bce_values, log_scale=False)
            plot_bce_vs_model_size(model_sizes, bce_values, log_scale=True)
            plot_power_law_fit(model_sizes, bce_values)
            plot_bce_distribution(bce_values, bins=10)
        except Exception as e:
            self.fail(f"Plot functions raised an exception: {e}")


class TestExperimentRunner(unittest.TestCase):
    def test_run_full_experiment(self):
        dummy_llm = DummyLLMInterface(fixed_value=0.1)
        history = "Test conversation."
        candidate1 = "Candidate1"
        candidate2 = "Candidate2"
        evidence = "Test evidence."
        results = run_full_experiment(history, candidate1, candidate2, evidence, dummy_llm)
        expected_keys = {"prior_prob_c1", "prior_prob_c2", "likelihood_c1", "likelihood_c2",
                         "posterior_prob_c1", "posterior_prob_c2", "BCE", "prompts"}
        self.assertTrue(expected_keys.issubset(results.keys()))


class TestPromptVariations(unittest.TestCase):
    def test_get_all_variations(self):
        history = "Test history."
        candidate = "Candidate"
        evidence = "Evidence."
        variations = get_all_variations(generate_likelihood_prompt, history, candidate, evidence, num_variations=5)
        self.assertEqual(len(variations), 5)

    def test_generate_all_prompt_variations(self):
        history = "Test history."
        candidate1 = "Candidate1"
        candidate2 = "Candidate2"
        evidence = "Evidence."
        variations_dict = generate_all_prompt_variations(history, candidate1, candidate2, evidence, num_variations=3)
        self.assertIn("prior", variations_dict)
        self.assertIn("likelihood_candidate1", variations_dict)
        self.assertIn("likelihood_candidate2", variations_dict)
        self.assertIn("posterior", variations_dict)

    def test_get_random_prompt_variation(self):
        prompts = ["Prompt1", "Prompt2", "Prompt3"]
        random_prompt = get_random_prompt_variation(prompts)
        self.assertIn(random_prompt, prompts)


class TestDataIO(unittest.TestCase):
    def test_save_and_load_results(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            temp_file = tmp.name

        sample_data = {"key": "value", "number": 123}
        save_results(sample_data, temp_file)
        loaded_data = load_results(temp_file)
        self.assertEqual(sample_data, loaded_data)
        os.remove(temp_file)


class TestPlotUtilities(unittest.TestCase):
    def test_plot_histogram(self):
        data = np.random.randn(100)
        try:
            plot_histogram(data, bins=20, title="Test Histogram", xlabel="X", ylabel="Frequency", show=False)
        except Exception as e:
            self.fail(f"plot_histogram raised an exception: {e}")

    def test_plot_scatter(self):
        x = np.random.rand(50)
        y = np.random.rand(50)
        try:
            plot_scatter(x, y, title="Test Scatter", xlabel="X", ylabel="Y", show=False, fit_line=True)
        except Exception as e:
            self.fail(f"plot_scatter raised an exception: {e}")

    def test_plot_line_chart(self):
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        try:
            plot_line_chart(x, y, title="Test Line Chart", xlabel="X", ylabel="sin(X)", show=False)
        except Exception as e:
            self.fail(f"plot_line_chart raised an exception: {e}")


class TestJsonData(unittest.TestCase):
    def test_json_data(self):
        try:
            with open('data/schema.json', 'r') as f_schema:
                schema = json.load(f_schema)

                try:
                    with open('data/data.json', 'r') as f_data:
                        data = json.load(f_data)
                except IOError:
                    print("could not read file:", f_data)

                for i in data["bayesian_reasoning"]: print(i["class_type"])

                try:
                    validate(instance = data, schema = schema)
                except ValidationError as ex_v:
                    print("validation error: ", ex_v)
                except SchemaError as ex_s:
                    print("schema error: ", ex_s)
                else:
                    print("valid json data")

        except IOError:
            print("could not read file:", f_schema)


if __name__ == '__main__':
    unittest.main()
