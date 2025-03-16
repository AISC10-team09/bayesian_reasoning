"""
This module provides a comprehensive suite of statistical and scaling analysis tools
for evaluating Bayesian Consistency Error and related metrics in LLM experiments.

Included functionalities:
    - Descriptive statistics (mean, median, std, etc.)
    - Confidence interval estimation (parametric and bootstrap)
    - One-sample t-test to check if BCE significantly differs from an ideal value (zero)
    - Linear regression analysis and power-law fitting (via log transformation)
    - Correlation analysis and ANOVA tests across multiple groups
    - Plotting utilities:
        * Histogram of BCE values with confidence intervals
        * Scatter and regression plots of BCE versus model size (linear and log-log scales)
        * Overlay of power-law fits over raw scaling data

These tools help in evaluating the consistency of Bayesian updates by LLMs and in analyzing
how BCE scales with model parameters (e.g., model size).
"""

import numpy as np
import scipy.stats as stats
import math
import matplotlib.pyplot as plt
import seaborn as sns


########################################
# Descriptive Statistics and Confidence Intervals
########################################

def compute_descriptive_stats(data):
    """
    Compute basic descriptive statistics for a dataset.

    Args:
        data (list or np.array): Numerical values (e.g., BCE values).

    Returns:
        dict: Contains mean, median, standard deviation, variance, min, max, and sample size.
    """
    data = np.array(data)
    stats_dict = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data, ddof=1),
        'variance': np.var(data, ddof=1),
        'min': np.min(data),
        'max': np.max(data),
        'n': len(data)
    }
    return stats_dict


def compute_confidence_interval(data, confidence=0.95):
    """
    Compute the confidence interval for the mean of the data using the t-distribution.

    Args:
        data (list or np.array): Numerical data.
        confidence (float, optional): Confidence level (default 0.95).

    Returns:
        tuple: (lower_bound, upper_bound)
    """
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    h = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean - h, mean + h


def bootstrap_confidence_interval(data, num_samples=1000, confidence=0.95, random_seed=None):
    """
    Compute a bootstrap confidence interval for the mean of the data.

    Args:
        data (list or np.array): Numerical data.
        num_samples (int, optional): Number of bootstrap samples.
        confidence (float, optional): Confidence level.
        random_seed (int, optional): Seed for reproducibility.

    Returns:
        tuple: (lower_bound, upper_bound)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    data = np.array(data)
    n = len(data)
    bootstrap_means = []
    for _ in range(num_samples):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    lower_bound = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    upper_bound = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)
    return lower_bound, upper_bound


def one_sample_ttest(data, hypothesized_mean=0.0):
    """
    Perform a one-sample t-test to assess if the mean of the data differs from a hypothesized value.

    Args:
        data (list or np.array): Numerical values (e.g., BCE values).
        hypothesized_mean (float, optional): The value to test against (default 0.0).

    Returns:
        tuple: (t_statistic, p_value)
    """
    data = np.array(data)
    t_stat, p_value = stats.ttest_1samp(data, hypothesized_mean)
    return t_stat, p_value


########################################
# Regression and Correlation Analysis
########################################

def perform_regression_analysis(x, y):
    """
    Perform linear regression analysis on independent variable x and dependent variable y.

    Args:
        x (list or np.array): Independent variable (e.g., model sizes).
        y (list or np.array): Dependent variable (e.g., BCE values).

    Returns:
        LinregressResult: Object containing slope, intercept, rvalue, pvalue, and standard error.
    """
    x = np.array(x)
    y = np.array(y)
    regression_result = stats.linregress(x, y)
    return regression_result


def fit_power_law(x, y):
    """
    Fit a power-law model y = a * x^b by performing linear regression on the log-transformed data.

    Args:
        x (list or np.array): Independent variable values (must be > 0).
        y (list or np.array): Dependent variable values (must be > 0).

    Returns:
        tuple: (a, b, regression_result) where:
            - a: scaling factor (intercept exponentiated)
            - b: power-law exponent
            - regression_result: full regression output from scipy.stats.linregress.
    """
    x = np.array(x)
    y = np.array(y)
    mask = (x > 0) & (y > 0)
    log_x = np.log(x[mask])
    log_y = np.log(y[mask])
    regression_result = stats.linregress(log_x, log_y)
    b = regression_result.slope
    a = np.exp(regression_result.intercept)
    return a, b, regression_result


def correlation_analysis(x, y, method='pearson'):
    """
    Compute the correlation between two datasets using Pearson or Spearman method.

    Args:
        x (list or np.array): First dataset.
        y (list or np.array): Second dataset.
        method (str, optional): 'pearson' (default) or 'spearman'.

    Returns:
        tuple: (correlation_coefficient, p_value)
    """
    x = np.array(x)
    y = np.array(y)
    if method == 'pearson':
        return stats.pearsonr(x, y)
    elif method == 'spearman':
        return stats.spearmanr(x, y)
    else:
        raise ValueError("Method must be either 'pearson' or 'spearman'.")


def anova_test(*groups):
    """
    Perform a one-way ANOVA test to compare means across multiple groups.

    Args:
        *groups: Variable number of arrays/lists, each representing a group.

    Returns:
        tuple: (F_statistic, p_value)
    """
    return stats.f_oneway(*groups)


########################################
# Scaling Analysis and Plotting Utilities
########################################

def plot_bce_vs_model_size(model_sizes, bce_values, log_scale=False,
                           title="BCE vs. Model Size", xlabel="Model Size", ylabel="BCE"):
    """
    Plot a scatter plot (with regression line) of BCE versus model sizes.
    Optionally use log-log scaling.

    Args:
        model_sizes (list or np.array): Model sizes (e.g., number of parameters).
        bce_values (list or np.array): Corresponding BCE values.
        log_scale (bool, optional): If True, plot on a log-log scale.
        title (str, optional): Plot title.
        xlabel (str, optional): Label for x-axis.
        ylabel (str, optional): Label for y-axis.
    """
    model_sizes = np.array(model_sizes)
    bce_values = np.array(bce_values)

    plt.figure(figsize=(8, 6))

    if log_scale:
        log_model_sizes = np.log(model_sizes)
        log_bce_values = np.log(bce_values)
        sns.regplot(x=log_model_sizes, y=log_bce_values, ci=95, scatter_kws={"s": 50}, line_kws={"color": "red"})
        plt.xlabel(f"log({xlabel})")
        plt.ylabel(f"log({ylabel})")
        plt.title(title + " (Log-Log Scale)")
    else:
        sns.regplot(x=model_sizes, y=bce_values, ci=95, scatter_kws={"s": 50}, line_kws={"color": "red"})
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

    plt.tight_layout()
    plt.show()


def fit_power_law_scaling(model_sizes, bce_values):
    """
    Fit a power-law model of the form BCE = a * model_size^(-b)
    using linear regression on log-transformed data.

    Args:
        model_sizes (list or np.array): Model sizes (must be > 0).
        bce_values (list or np.array): BCE values (must be > 0).

    Returns:
        tuple: (a, b, regression_result)
            where a is the scaling factor, b is the exponent, and regression_result contains full regression output.
    """
    model_sizes = np.array(model_sizes)
    bce_values = np.array(bce_values)
    mask = (model_sizes > 0) & (bce_values > 0)
    log_x = np.log(model_sizes[mask])
    log_y = np.log(bce_values[mask])

    regression_result = stats.linregress(log_x, log_y)
    # For BCE expected to decrease with size, we may represent as: BCE = a * model_size^(-b)
    b = -regression_result.slope
    a = np.exp(regression_result.intercept)
    return a, b, regression_result


def plot_power_law_fit(model_sizes, bce_values, title="Power Law Fit for BCE Scaling",
                       xlabel="Model Size", ylabel="BCE"):
    """
    Plot the raw BCE data along with the power-law fit curve.

    Args:
        model_sizes (list or np.array): Model sizes.
        bce_values (list or np.array): BCE values.
        title (str, optional): Plot title.
        xlabel (str, optional): x-axis label.
        ylabel (str, optional): y-axis label.
    """
    model_sizes = np.array(model_sizes)
    bce_values = np.array(bce_values)

    a, b, _ = fit_power_law_scaling(model_sizes, bce_values)

    x_range = np.linspace(np.min(model_sizes), np.max(model_sizes), 100)
    fitted_bce = a * x_range ** (-b)

    plt.figure(figsize=(8, 6))
    plt.scatter(model_sizes, bce_values, color='blue', s=50, label='Data')
    plt.plot(x_range, fitted_bce, color='red', linewidth=2,
             label=f'Fit: BCE = {a:.3f} * x^(-{b:.3f})')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_bce_distribution(bce_values, bins=30, title="BCE Distribution"):
    """
    Plot a histogram of BCE values and annotate the plot with the mean and 95% confidence interval.

    Args:
        bce_values (list or np.array): BCE values.
        bins (int, optional): Number of bins for the histogram.
        title (str, optional): Plot title.
    """
    bce_array = np.array(bce_values)
    plt.figure(figsize=(8, 5))
    plt.hist(bce_array, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel("BCE")
    plt.ylabel("Frequency")
    plt.title(title)
    mean = np.mean(bce_array)
    ci_low, ci_high = compute_confidence_interval(bce_array)
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f"Mean: {mean:.4f}")
    plt.axvline(ci_low, color='green', linestyle='dotted', linewidth=2, label=f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    plt.axvline(ci_high, color='green', linestyle='dotted', linewidth=2)
    plt.legend()
    plt.tight_layout()
    plt.show()


########################################
# __main__ - Example Usage and Testing
########################################

if __name__ == "__main__":
    # Synthetic example data:
    # BCE values from experiments
    example_bce = [0.12, 0.15, 0.09, 0.11, 0.14, 0.13, 0.16, 0.10, 0.08, 0.12]

    # Model sizes (e.g., in millions of parameters) and corresponding BCE values
    model_sizes = [100, 200, 300, 400, 500]
    bce_scaling = [0.20, 0.15, 0.12, 0.10, 0.08]

    # DESCRIPTIVE STATISTICS & CONFIDENCE INTERVALS
    stats_desc = compute_descriptive_stats(example_bce)
    print("Descriptive Statistics for BCE:")
    print(stats_desc)

    ci = compute_confidence_interval(example_bce)
    print("\n95% Confidence Interval for BCE Mean (Parametric):", ci)

    boot_ci = bootstrap_confidence_interval(example_bce, num_samples=1000, random_seed=42)
    print("Bootstrap 95% Confidence Interval for BCE Mean:", boot_ci)

    # ONE-SAMPLE T-TEST
    t_stat, p_val = one_sample_ttest(example_bce, hypothesized_mean=0.0)
    print(f"\nOne-Sample t-test for BCE (H0: mean = 0): t = {t_stat:.4f}, p = {p_val:.4f}")

    # REGRESSION ANALYSIS
    reg_result = perform_regression_analysis(model_sizes, bce_scaling)
    print("\nLinear Regression for BCE Scaling:")
    print(f"Slope: {reg_result.slope:.4f}, Intercept: {reg_result.intercept:.4f}, R-value: {reg_result.rvalue:.4f}")

    # POWER-LAW FITTING
    a, b, pl_reg = fit_power_law_scaling(model_sizes, bce_scaling)
    print("\nPower-Law Regression for BCE Scaling:")
    print(f"Scaling factor (a): {a:.4f}, Exponent (b): {b:.4f}")

    # CORRELATION ANALYSIS (example with synthetic KL divergence values)
    kl_values = [0.50, 0.45, 0.40, 0.35, 0.30]
    corr_coef, corr_p = correlation_analysis(bce_scaling, kl_values, method='pearson')
    print("\nCorrelation between BCE and KL Divergence:")
    print(f"Correlation Coefficient: {corr_coef:.4f}, p-value: {corr_p:.4f}")

    # ANOVA TEST (comparing three synthetic groups)
    group1 = [0.20, 0.22, 0.19, 0.21, 0.20]
    group2 = [0.15, 0.16, 0.14, 0.15, 0.16]
    group3 = [0.12, 0.11, 0.13, 0.12, 0.12]
    f_stat, anova_p = anova_test(group1, group2, group3)
    print("\nANOVA Test across three groups:")
    print(f"F-statistic: {f_stat:.4f}, p-value: {anova_p:.4f}")

    # PLOTTING UTILITIES
    plot_bce_distribution(example_bce, bins=10, title="Example BCE Distribution")
    plot_bce_vs_model_size(model_sizes, bce_scaling, log_scale=False,
                           title="BCE vs. Model Size (Linear Scale)",
                           xlabel="Model Size (Millions)", ylabel="BCE")
    plot_bce_vs_model_size(model_sizes, bce_scaling, log_scale=True,
                           title="BCE vs. Model Size (Log-Log Scale)",
                           xlabel="Model Size (Millions)", ylabel="BCE")
    plot_power_law_fit(model_sizes, bce_scaling,
                       title="Power Law Fit for BCE Scaling",
                       xlabel="Model Size (Millions)", ylabel="BCE")
