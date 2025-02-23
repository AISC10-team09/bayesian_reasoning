"""
plot.py

This module provides generic plotting utilities for the project. It includes functions to:
  - Set default plotting styles.
  - Save the current plot to a file.
  - Plot histograms, scatter plots, and line charts with standard aesthetics.
  - Optionally display and close plots.

These functions can be used by various parts of the codebase (e.g., scaling analysis, result visualization).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def set_default_style():
    """
    Set default style for matplotlib and seaborn plots.
    """
    sns.set(style="whitegrid")
    plt.rcParams.update({
        "figure.figsize": (8, 6),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def save_plot(save_path, dpi=300, bbox_inches="tight"):
    """
    Save the current matplotlib figure to a file.

    Args:
        save_path (str): Path where the figure will be saved.
        dpi (int, optional): Dots per inch (default 300).
        bbox_inches (str, optional): Bounding box argument for saving (default "tight").
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Plot saved to {save_path}")


def show_and_close(show=True):
    """
    Show the current plot if desired and then close it.

    Args:
        show (bool, optional): Whether to display the plot (default True).
    """
    if show:
        plt.show()
    plt.close()


def plot_histogram(data, bins=30, title="Histogram", xlabel="Value", ylabel="Frequency",
                   show=True, save_path=None):
    """
    Plot a histogram of the provided data.

    Args:
        data (list or np.array): Data values to plot.
        bins (int, optional): Number of histogram bins (default 30).
        title (str, optional): Title of the plot.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        show (bool, optional): Whether to display the plot (default True).
        save_path (str, optional): If provided, the plot is saved to this path.
    """
    set_default_style()
    plt.figure()
    plt.hist(data, bins=bins, color="skyblue", edgecolor="black", alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path:
        save_plot(save_path)
    show_and_close(show)


def plot_scatter(x, y, title="Scatter Plot", xlabel="X", ylabel="Y", show=True, save_path=None, fit_line=False):
    """
    Plot a scatter plot of x versus y with an optional linear fit.

    Args:
        x (list or np.array): Data for the x-axis.
        y (list or np.array): Data for the y-axis.
        title (str, optional): Title of the plot.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        show (bool, optional): Whether to display the plot (default True).
        save_path (str, optional): If provided, the plot is saved to this path.
        fit_line (bool, optional): If True, a linear regression line is added to the plot.
    """
    set_default_style()
    plt.figure()
    plt.scatter(x, y, color="blue", alpha=0.6, edgecolor="k")
    if fit_line:
        # Fit a linear regression line to the data.
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m * np.array(x) + b, color="red", linestyle="--", label=f"Fit: y={m:.2f}x+{b:.2f}")
        plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path:
        save_plot(save_path)
    show_and_close(show)


def plot_line_chart(x, y, title="Line Chart", xlabel="X", ylabel="Y", show=True, save_path=None):
    """
    Plot a line chart for the given x and y data.

    Args:
        x (list or np.array): Data for the x-axis.
        y (list or np.array): Data for the y-axis.
        title (str, optional): Title of the chart.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        show (bool, optional): Whether to display the chart (default True).
        save_path (str, optional): If provided, the chart is saved to this path.
    """
    set_default_style()
    plt.figure()
    plt.plot(x, y, marker="o", color="green")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path:
        save_plot(save_path)
    show_and_close(show)


if __name__ == "__main__":
    # Example usage of the plotting utilities.

    # Histogram example.
    data = np.random.randn(1000)
    plot_histogram(data, bins=20, title="Random Data Histogram", xlabel="Value", ylabel="Frequency", show=True)

    # Line chart example.
    x = np.linspace(0, 10, 50)
    y = np.sin(x)
    plot_line_chart(x, y, title="Sine Wave", xlabel="X", ylabel="sin(X)", show=True)

    # Scatter plot example with regression line.
    x_scatter = np.random.rand(50) * 10
    y_scatter = 2 * x_scatter + np.random.randn(50)
    plot_scatter(x_scatter, y_scatter, title="Scatter Plot with Fit", xlabel="X", ylabel="Y", show=True, fit_line=True)
