import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
from src.metrics import single_evidence_estimate, pairwise_mse_of_group

# Load the Pythia data
df = pd.read_csv('data/pythia_logprobs.csv')

# Extract model size from model_name
def extract_model_size(model_name):
    # Look for sizes like 160m, 1b, 6.9b, 12b in model names
    size_match = re.search(r'pythia-(\d+\.?\d*[mb])', model_name.lower())
    if size_match:
        return size_match.group(1)
    return 'unknown'

# Extract training step from model_kwargs
def extract_step(kwargs_str):
    try:
        # Parse the kwargs string into a dict
        kwargs = json.loads(kwargs_str.replace("'", '"'))
        revision = kwargs.get('revision', '')
        # Extract the step number
        step_match = re.search(r'step(\d+)', revision)
        if step_match:
            return int(step_match.group(1))
        return 0
    except:
        return 0

# Add model size and step columns
df['model_size'] = df['model_name'].apply(extract_model_size)
df['step'] = df['model_kwargs'].apply(extract_step)

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

# Group by model_size and step and calculate all pairwise Bayesian Consistency Errors
pairs_data = []
for (size, step), group in df.groupby(['model_size', 'step']):
    # Get all pairwise square differences
    bce_pairs = pairwise_mse_of_group(group, 'evidence_estimate')
    
    # Add each individual comparison to the data
    for bce_value in bce_pairs:
        pairs_data.append({
            'model_size': size,
            'step': step,
            'BCE': bce_value
        })

pairs_df = pd.DataFrame(pairs_data)

# Calculate means for a result table
result_df = pairs_df.groupby(['model_size', 'step'])['BCE'].mean().reset_index()

# Sort model sizes in a logical order (smallest to largest)
def size_sorter(size_str):
    if 'm' in size_str:
        return float(size_str.replace('m', '')) / 1000
    elif 'b' in size_str:
        return float(size_str.replace('b', ''))
    return 0

model_sizes = sorted(df['model_size'].unique(), key=size_sorter)

# Create faceted box plots
fig, axes = plt.subplots(1, len(model_sizes), figsize=(16, 6), sharey=True)

# Create a plot for each model size
for i, size in enumerate(model_sizes):
    # Filter data for this size
    size_data = pairs_df[pairs_df['model_size'] == size]
    
    # Create boxplot
    ax = axes[i] if len(model_sizes) > 1 else axes
    sns.boxplot(x='step', y='BCE', data=size_data, ax=ax, showfliers=False)
    
    # Set title and format
    ax.set_title(f"Pythia-{size}")
    ax.set_xlabel('Training Steps')
    
    # Get step counts
    step_counts = size_data.groupby('step').size()
    steps = sorted(step_counts.index)
    
    # Create labels with counts
    new_labels = [f"{step}\n(n={step_counts[step]})" for step in steps]
    ax.set_xticklabels(new_labels)
    
    # Only show y-axis label on the leftmost plot
    if i > 0:
        ax.set_ylabel('')

# Set common y-label
if len(model_sizes) > 1:
    axes[0].set_ylabel('Pairwise Bayesian Consistency Error')
else:
    axes.set_ylabel('Pairwise Bayesian Consistency Error')

plt.suptitle('Pythia Models: BCE by Model Size and Training Steps', fontsize=16, y=1.05)
plt.tight_layout()
plt.savefig('results/plots/pythia_bce_scaling_steps.png', dpi=300)

# Print the results table
print("Average BCE by Model Size and Training Step:")
print(result_df.sort_values(['model_size', 'step']))

# Also create a line plot showing trends
plt.figure(figsize=(10, 6))
sns.lineplot(data=result_df, x='step', y='BCE', hue='model_size', marker='o')
plt.title('Pythia Models: BCE Trends by Model Size and Training Steps')
plt.xlabel('Training Steps')
plt.ylabel('Average BCE')
plt.tight_layout()
plt.savefig('results/plots/pythia_bce_scaling_trends.png', dpi=300)