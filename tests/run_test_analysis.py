"""
Simple script to test analysis of logprobs from the small test dataset.
"""

import pandas as pd
import numpy as np
from src.analyzer import Analyzer
from src.metrics import single_evidence_estimate, pairwise_bce_of_group

# Load the collected logprobs
df = pd.read_csv('tests/data/test_logprobs.csv')
print(f"Loaded {len(df)} rows of logprobs data")

# Initialize analyzer
analyzer = Analyzer(df)
print("Analyzer initialized")

# Calculate evidence estimates
analyzer = analyzer.add_column(
    "evidence_estimate",
    lambda df: single_evidence_estimate(df, "prior_logprob", "likelihood_logprob", "posterior_logprob")
)
print("Evidence estimates calculated:")
print(analyzer.df[["class", "evidence_estimate"]].to_string())

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
print("\nBayesian Consistency Error (BCE) results:")
print(bce_df.to_string())

# Print out the calculation that contributed to BCE for better understanding
print("\nDetailed BCE calculation:")
for i in range(len(analyzer.df)):
    row = analyzer.df.iloc[i]
    prior = row["prior_logprob"]
    like = row["likelihood_logprob"]
    post = row["posterior_logprob"]
    estimate = row["evidence_estimate"]
    print(f"Class: {row['class']}")
    print(f"  Prior: {prior:.2f}, Likelihood: {like:.2f}, Posterior: {post:.2f}")
    print(f"  Evidence estimate: {prior:.2f} + {like:.2f} - {post:.2f} = {estimate:.2f}")