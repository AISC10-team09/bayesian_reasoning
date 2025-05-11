"""
Simple script to test visualization of logprobs analysis results.
"""

import pandas as pd
import numpy as np
import altair as alt
from src.analyzer import Analyzer
from src.metrics import single_evidence_estimate, pairwise_bce_of_group
from src.visualizer import VisualisationConfig, visualize
import os

# Load the collected logprobs
df = pd.read_csv('tests/data/test_small_logprobs.csv')
print(f"Loaded {len(df)} rows of logprobs data")

# Initialize analyzer
analyzer = Analyzer(df)

# Calculate evidence estimates
analyzer = analyzer.add_column(
    "evidence_estimate",
    lambda df: single_evidence_estimate(df, "prior_logprob", "likelihood_logprob", "posterior_logprob")
)

# Create a comparison DataFrame for visualization
comparison_df = pd.DataFrame({
    "component": ["prior", "posterior", "evidence_estimate", "likelihood"] * len(analyzer.df),
    "value": np.concatenate([
        analyzer.df["prior_logprob"].values,
        analyzer.df["posterior_logprob"].values,
        analyzer.df["evidence_estimate"].values,
        analyzer.df["likelihood_logprob"].values
    ]),
    "class": np.repeat(analyzer.df["class"].values, 4)
})

# Ensure output directory exists
os.makedirs("results/small_test", exist_ok=True)

# Create visualization of log probabilities
config = VisualisationConfig(
    plot_fn=alt.Chart.mark_bar,
    x_category="component",
    y_category="value",
    color_category="class",
    fig_title="Log Probabilities by Component and Class"
)

chart = visualize(comparison_df, config)
chart.save("results/small_test/log_probs_comparison.html")
print("Saved log probabilities visualization to results/small_test/log_probs_comparison.html")

# Calculate BCE for each class_type and evidence combination
bce_data = []
for (class_type, evidence_text), group in analyzer.df.groupby(["class_type", "evidence_text"]):
    # Skip groups with less than 2 items
    if len(group) < 2:
        continue
        
    # Get BCE values
    bce_values = pairwise_bce_of_group(
        group, "prior_logprob", "likelihood_logprob", "posterior_logprob", square=True
    )
    
    # Calculate mean BCE
    mean_bce = np.mean(bce_values) if len(bce_values) > 0 else np.nan
    
    # Store results
    bce_data.append({
        "class_type": class_type,
        "evidence_text": evidence_text,
        "mean_bce": mean_bce
    })

# Create BCE DataFrame
bce_df = pd.DataFrame(bce_data)

# Create BCE visualization
if not bce_df.empty:
    bce_config = VisualisationConfig(
        plot_fn=alt.Chart.mark_bar,
        x_category="class_type",
        y_category="mean_bce",
        fig_title="Bayesian Consistency Error by Class Type"
    )
    bce_chart = visualize(bce_df, bce_config)
    bce_chart.save("results/small_test/bce_by_class_type.html")
    print("Saved BCE visualization to results/small_test/bce_by_class_type.html")
else:
    print("No BCE data available for visualization (need at least two classes per group)")

# Print completion message
print("\nAnalysis and visualization complete. Explore the results in the results/small_test directory.")