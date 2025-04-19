import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from src.metrics import single_evidence_estimate, pairwise_mse_of_group

# Load the data
df = pd.read_csv('data/logprobs.csv')

# Extract model family from model_name
def extract_model_family(model_name):
    if 'Llama' in model_name:
        return 'Llama'
    elif 'claude' in model_name.lower():
        return 'Claude'
    elif 'gpt' in model_name.lower():
        return 'GPT'
    else:
        return 'Other'

# Extract model size from model_name
def extract_model_size(model_name):
    # Try to find a number followed by B (for billion parameters)
    size_match = re.search(r'(\d+\.?\d*)B', model_name)
    if size_match:
        return float(size_match.group(1))
    return np.nan

# Add model family and size columns
df['model_family'] = df['model_name'].apply(extract_model_family)
df['model_size'] = df['model_name'].apply(extract_model_size)

# Calculate evidence estimates for each row
df['evidence_estimate'] = df.apply(
    lambda row: single_evidence_estimate(
        row, 
        'prior_logprob', 
        'likelihood_logprob', 
        'posterior_logprob'
    ), 
    axis=1
)

# Group by model_name and calculate all pairwise Bayesian Consistency Errors (not just the mean)
pairs_data = []
for model, group in df.groupby('model_name'):
    # Get all pairwise square differences instead of just the mean
    bce_pairs = pairwise_mse_of_group(group, 'evidence_estimate')
    family = group['model_family'].iloc[0]
    size = group['model_size'].iloc[0]
    
    # Add each individual comparison to the data
    for bce_value in bce_pairs:
        pairs_data.append({
            'model_name': model,
            'model_family': family,
            'model_size': size,
            'BCE': bce_value
        })

pairs_df = pd.DataFrame(pairs_data)

# Calculate means for a separate result table
result_df = pairs_df.groupby(['model_name', 'model_family', 'model_size'])['BCE'].mean().reset_index()

# Sort by model size within each family
pairs_df = pairs_df.sort_values(['model_family', 'model_size'])
result_df = result_df.sort_values(['model_family', 'model_size'])

# Create the box plot with improved styling
plt.figure(figsize=(14, 8))

# Create separate figure with subplots instead of FacetGrid for more control
fig, axes = plt.subplots(1, len(df['model_family'].unique()), figsize=(14, 6), sharey=True)

# Get unique model families
model_families = df['model_family'].unique()

# Create a plot for each model family
for i, family in enumerate(model_families):
    # Filter data for this family
    family_data = pairs_df[pairs_df['model_family'] == family]
    
    # Create boxplot without outliers
    sns.boxplot(x='model_name', y='BCE', data=family_data, ax=axes[i], showfliers=False)
    
    # Set title and format
    axes[i].set_title(family)
    axes[i].set_xlabel('')
    
    # Get model names and counts
    model_counts = family_data.groupby('model_name').size()
    model_names = model_counts.index
    
    # Create new labels with counts
    new_labels = [f"{name}\n(n={model_counts[name]})" for name in model_names]
    
    # Rotate labels diagonally
    axes[i].set_xticklabels(new_labels, rotation=45, ha='right')
    
    # Only show y-axis label on the leftmost plot
    if i > 0:
        axes[i].set_ylabel('')

# Set common y-label
axes[0].set_ylabel('Pairwise Bayesian Consistency Error')

plt.tight_layout()
plt.savefig('results/plots/model_bce_scaling_boxplot.png', dpi=300)
plt.show()

# Print the results table
print(result_df.sort_values('BCE'))
