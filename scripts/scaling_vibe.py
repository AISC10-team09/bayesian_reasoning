import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from src.metrics import single_evidence_estimate, pairwise_error_of_group
from matplotlib.ticker import ScalarFormatter, FuncFormatter

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

# Group by model_name, class_type, AND evidence_text to calculate pairwise BCE values
pairs_data = []
for (model, class_type, evidence), group in df.groupby(['model_name', 'class_type', 'evidence_text']):
    # Skip groups with less than 2 items (can't compute pairwise differences)
    if len(group) < 2:
        continue
        
    # Get all pairwise square differences within this group
    bce_pairs = pairwise_error_of_group(group, 'evidence_estimate')
    family = group['model_family'].iloc[0]
    size = group['model_size'].iloc[0]
    
    # Add each individual comparison to the data
    for bce_value in bce_pairs:
        pairs_data.append({
            'model_name': model,
            'model_family': family,
            'model_size': size,
            'class_type': class_type,
            'evidence_text': evidence,
            'BCE': bce_value
        })

pairs_df = pd.DataFrame(pairs_data)

# Calculate means and medians for result tables - now including evidence_text
mean_df = pairs_df.groupby(['model_name', 'model_family', 'model_size'])['BCE'].mean().reset_index()
median_df = pairs_df.groupby(['model_name', 'model_family', 'model_size'])['BCE'].median().reset_index()

# Also create versions that include evidence_text
mean_with_evidence_df = pairs_df.groupby(['model_name', 'model_family', 'model_size', 'evidence_text'])['BCE'].mean().reset_index()
median_with_evidence_df = pairs_df.groupby(['model_name', 'model_family', 'model_size', 'evidence_text'])['BCE'].median().reset_index()

# Sort by model size within each family
pairs_df = pairs_df.sort_values(['model_family', 'model_size'])
mean_df = mean_df.sort_values(['model_family', 'model_size'])
median_df = median_df.sort_values(['model_family', 'model_size'])

# Create box plots with improved styling for each model family
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

# Create scaling plots with parameter size on x-axis - MEAN
plt.figure(figsize=(10, 6))

# Create a line plot for each model family - MEAN
for family in model_families:
    # Filter data for this family
    family_mean = mean_df[mean_df['model_family'] == family]
    
    # Skip if no data for this family
    if family_mean.empty:
        continue
    
    # Plot the line with markers
    plt.plot(family_mean['model_size'], family_mean['BCE'], 
             marker='o', linestyle='-', label=family)

plt.xscale('log')  # Use log scale for parameter size
plt.xlabel('Model Size (Billion Parameters, log scale)')
plt.ylabel('Mean Pairwise Bayesian Consistency Error')
plt.title('Mean BCE vs. Model Size (Grouped by Model Family)')
plt.grid(True, alpha=0.3)
plt.legend()

# Fix the x-axis scaling notation to properly show billions
def billions_formatter(x, pos):
    """Format x as a number in billions."""
    return f'{x:.0f}'

plt.gca().xaxis.set_major_formatter(FuncFormatter(billions_formatter))
plt.tight_layout()
plt.savefig('results/plots/model_bce_scaling_mean.png', dpi=300)

# Create scaling plots with parameter size on x-axis - MEDIAN
plt.figure(figsize=(10, 6))

# Create a line plot for each model family - MEDIAN
for family in model_families:
    # Filter data for this family
    family_median = median_df[median_df['model_family'] == family]
    
    # Skip if no data for this family
    if family_median.empty:
        continue
    
    # Plot the line with markers
    plt.plot(family_median['model_size'], family_median['BCE'], 
             marker='o', linestyle='-', label=family)

plt.xscale('log')  # Use log scale for parameter size
plt.xlabel('Model Size (Billion Parameters, log scale)')
plt.ylabel('Median Pairwise Bayesian Consistency Error')
plt.title('Median BCE vs. Model Size (Grouped by Model Family)')
plt.grid(True, alpha=0.3)
plt.legend()

# Fix the x-axis scaling notation to properly show billions
plt.gca().xaxis.set_major_formatter(FuncFormatter(billions_formatter))
plt.tight_layout()
plt.savefig('results/plots/model_bce_scaling_median.png', dpi=300)

# Print the results tables
print("Mean BCE by Model:")
print(mean_df.sort_values(['model_family', 'model_size']))

print("\nMedian BCE by Model:")
print(median_df.sort_values(['model_family', 'model_size']))

# Filter for only the specified models for class_type analysis
# You can modify this list as needed
filtered_df = df[df['model_name'].isin(['meta-llama/Llama-3.1-8B', 'openai-community/gpt2-xl'])]

# Group by model_name, class_type, AND evidence_text to calculate pairwise BCE values
class_pairs_data = []
for (model, class_type, evidence), group in filtered_df.groupby(['model_name', 'class_type', 'evidence_text']):
    # Skip groups with less than 2 items
    if len(group) < 2:
        continue
        
    # Get all pairwise square differences
    bce_pairs = pairwise_error_of_group(group, 'evidence_estimate')
    family = group['model_family'].iloc[0]
    size = group['model_size'].iloc[0]
    
    # Add each individual comparison to the data
    for bce_value in bce_pairs:
        class_pairs_data.append({
            'model_name': model,
            'class_type': class_type,
            'evidence_text': evidence,
            'model_family': family,
            'model_size': size,
            'BCE': bce_value
        })

class_pairs_df = pd.DataFrame(class_pairs_data)

# Calculate means for the result table with evidence_text
class_result_df = class_pairs_df.groupby(['model_name', 'class_type', 'model_family', 'model_size'])['BCE'].mean().reset_index()
class_result_with_evidence_df = class_pairs_df.groupby(['model_name', 'class_type', 'evidence_text', 'model_family', 'model_size'])['BCE'].mean().reset_index()

# Create the box plot with improved styling
plt.figure(figsize=(14, 8))

# Create separate figure with subplots for each model family
fig, axes = plt.subplots(1, len(filtered_df['model_family'].unique()), figsize=(14, 6), sharey=True)

# Get unique model families from filtered data
filtered_model_families = filtered_df['model_family'].unique()

# Check if we need to convert axes to list if there's only one subplot
if len(filtered_model_families) == 1:
    axes = [axes]

# Create a plot for each model family
for i, family in enumerate(filtered_model_families):
    # Filter data for this family
    family_data = class_pairs_df[class_pairs_df['model_family'] == family]
    
    # Create boxplot without outliers
    sns.boxplot(x='class_type', y='BCE', data=family_data, ax=axes[i], showfliers=False)
    
    # Set title and format
    axes[i].set_title(f"{family} ({family_data['model_name'].iloc[0]})")
    axes[i].set_xlabel('')
    
    # Get class types and counts
    class_counts = family_data.groupby('class_type').size()
    class_names = class_counts.index
    
    # Create new labels with counts
    new_labels = [f"{name}\n(n={class_counts[name]})" for name in class_names]
    
    # Rotate labels diagonally
    axes[i].set_xticklabels(new_labels, rotation=45, ha='right')
    
    # Only show y-axis label on the leftmost plot
    if i > 0:
        axes[i].set_ylabel('')

# Set common y-label
axes[0].set_ylabel('Pairwise Bayesian Consistency Error')

plt.tight_layout()
plt.savefig('results/plots/class_type_bce_boxplot.png', dpi=300)

# Print the class type results table
print("\nBCE by Model, Class Type, and Evidence Text:")
print(class_result_with_evidence_df.sort_values(['model_family', 'model_size', 'class_type', 'evidence_text']))

# Print additional tables with evidence_text included
print("\nMean BCE by Model and Evidence Text:")
print(mean_with_evidence_df.sort_values(['model_family', 'model_size', 'evidence_text']))

print("\nMedian BCE by Model and Evidence Text:")
print(median_with_evidence_df.sort_values(['model_family', 'model_size', 'evidence_text']))


