import numpy as np
import pandas as pd
import altair as alt
from typing import List, Optional, Callable, Dict, Any, Union, Tuple
from dataclasses import dataclass, field


@dataclass
class AnalysisConfig:
    """Configuration for the LogprobAnalyzer visualization."""

    plot_fn: Callable[..., alt.Chart] = alt.Chart.mark_point  # Default to point plot
    fig_title: Optional[str] = None
    plot_fn_kwargs: Dict[str, Any] = field(default_factory=dict)
    chart_properties: Dict[str, Any] = field(default_factory=dict)
    x_category: Optional[str] = None  # Column for X-axis
    y_category: Optional[str] = (
        None  # Column for Y-axis (should match metric_name used in Analyzer)
    )
    x_offset_category: Optional[str] = None
    y_offset_category: Optional[str] = None
    color_category: Optional[str] = None
    color_scheme: Optional[str] = None
    color_domain: Optional[List[str]] = None
    color_range: Optional[List[str]] = None
    color_legend: Optional[alt.Legend] = None
    opacity_category: Optional[str] = None
    opacity_legend: Optional[alt.Legend] = None
    size_category: Optional[str] = None  # Added for scatter plots etc.
    size_legend: Optional[alt.Legend] = None
    shape_category: Optional[str] = None  # Added for different shapes
    shape_legend: Optional[alt.Legend] = None
    layer_category: Optional[str] = None
    facet_category: Optional[str] = None
    facet_columns: Optional[int] = None
    h_concat_category: Optional[str] = None
    v_concat_category: Optional[str] = None
    shared_y_scale: bool = False
    tooltip_fields: Optional[List[alt.Tooltip]] = None
    titles: Optional[Dict[str, str]] = field(
        default_factory=dict
    )  # Titles for axes/legends
    legend_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "orient": "bottom",
            "columns": 3,
            "titleAlign": "center",
            "labelLimit": 1000,
        }
    )
    interactive_chart: bool = False  # Enable zooming and panning


# Assuming AnalysisConfig is defined in analysis_config.py or above
# from analysis_config import AnalysisConfig


class LogprobAnalyzer:
    """
    Analyzes and visualizes log probability data using Altair charts.

    Usage flow:
    1. Initialize with data source
    2. Add filters with add_filter()
    3. Add categories with add_categorizer()
    4. Apply sorting/renaming with set_sort_order() and add_rename_mapping()
    5. Process data with process_dataframe()
    6. Calculate metrics with calculate_metric()
    7. Visualize with visualize()
    """

    def __init__(self, source: Union[pd.DataFrame, str]):
        """Initialize the LogprobAnalyzer with a DataFrame or CSV path."""
        if isinstance(source, str):
            try:
                self.original_df = pd.read_csv(source)
            except FileNotFoundError:
                raise FileNotFoundError(f"CSV file not found at path: {source}")
            except Exception as e:
                raise ValueError(f"Error reading CSV file: {e}")
        elif isinstance(source, pd.DataFrame):
            self.original_df = source.copy()
        else:
            raise ValueError("source must be a pandas DataFrame or a file path string.")

        # Store processing steps added by the user
        self._filters = []
        self._categorizers = []
        self._rename_mappings = {}
        self._sort_orders = {}

        # Store the current state of the DataFrame
        self.current_df = self.original_df.copy()

        # Store calculated metrics
        self.metric_df = None

    def add_filter(self, filter_spec: Union[Dict[str, List], Callable[[pd.DataFrame], pd.DataFrame]]):
        """
        Adds a filter to the processing pipeline.
        
        Args:
            filter_spec: Either:
                - A dictionary mapping columns to allowed values: {"column": [val1, val2]}
                - A callable that takes a DataFrame and returns a filtered DataFrame
        """
        if callable(filter_spec):
            self._filters.append(filter_spec)
        elif isinstance(filter_spec, dict):
            def dict_filter(df):
                result = df.copy()
                for col, values in filter_spec.items():
                    if col in result.columns:
                        result = result[result[col].isin(values)]
                return result
            self._filters.append(dict_filter)
        else:
            raise ValueError("filter_spec must be either a dictionary or a callable function.")
        self.process_dataframe()  # Automatically process after adding filter
        return self  # Allow chaining

    def add_category(
        self,
        output_columns: Union[str, List[str]],
        categorizer: Callable,
        source_columns: Union[str, Tuple[str, ...]],
    ):
        """Adds a categorizer function to create new columns."""
        if not callable(categorizer):
            raise ValueError("categorizer must be a callable function.")
        if isinstance(source_columns, list):
            source_columns = tuple(source_columns)
        if not isinstance(output_columns, (str, list)):
            raise ValueError("output_columns must be a string or a list of strings.")
        if isinstance(output_columns, list) and not all(
            isinstance(c, str) for c in output_columns
        ):
            raise ValueError(
                "If output_columns is a list, all elements must be strings."
            )

        self._categorizers.append((output_columns, source_columns, categorizer))
        self.process_dataframe()
        return self

    def add_rename_mapping(self, column: str, mapping: Dict[str, str]):
        """Adds rename mappings for values within a specific column."""
        if column not in self._rename_mappings:
            self._rename_mappings[column] = {}
        self._rename_mappings[column].update(mapping)
        self.process_dataframe()
        return self

    def set_sort_order(self, sort_spec: Union[Dict[str, List[str]], Callable[[pd.DataFrame], pd.DataFrame]]):
        """
        Sets custom sort order for categorical columns or applies a custom sorting function.
        
        Args:
            sort_spec: Either:
                - A dictionary mapping columns to their sort order: {"column": [val1, val2]}
                - A callable that takes a DataFrame and returns a sorted DataFrame
        """
        if callable(sort_spec):
            # Store callables separately from column-based sort orders
            if not hasattr(self, '_sort_functions'):
                self._sort_functions = []
            self._sort_functions.append(sort_spec)
        elif isinstance(sort_spec, dict):
            # Update existing sort orders dict
            self._sort_orders.update(sort_spec)
        else:
            raise ValueError("sort_spec must be either a dictionary or a callable function.")
        self.process_dataframe()  # Automatically process after setting sort order
        return self

    def reset_processing(self):
        """Resets all added filters, categorizers, renames, and sorts."""
        self._filters = []
        self._categorizers = []
        self._rename_mappings = {}
        self._sort_orders = {}
        self.current_df = self.original_df.copy()
        self.metric_df = None
        print("Processing steps reset. DataFrame reverted to original.")
        return self

    def process_dataframe(self) -> pd.DataFrame:
        """
        Applies all stored filters, categorizers, renames, and sorts.
        Updates and returns self.current_df with the result.
        """
        df = self.original_df.copy()  # Always start fresh

        # Apply filters
        for i, f in enumerate(self._filters):
            try:
                df = f(df)
            except Exception as e:
                print(
                    f"Warning: Error applying filter #{i + 1} ({getattr(f, '__name__', 'anonymous')}): {e}. Skipping filter."
                )

        # Apply categorizers
        df = self._apply_categorizers(df)

        # Apply renaming and sorting
        df = self._apply_renaming_and_sorting(df)

        self.current_df = df
        return self.current_df

    def calculate_metric(
        self,
        metric: Callable,
        metric_name: str,
        group_by_cols: List[str] | None = None,
        metric_kwargs: Dict | None = None,
    ) -> pd.DataFrame:
        """
        Calculates metrics on the processed data, with explicit grouping.

        Args:
            metric: Function to calculate the metric
            metric_name: Name for the calculated metric column
            group_by_cols: Columns to group by (None for row-wise calculation)
            metric_kwargs: Additional arguments for the metric function

        Returns:
            DataFrame with calculated metrics
        """
        if self.current_df is None or self.current_df.empty:
            print("Error: No processed data available. Call process_dataframe() first.")
            return None

        df_copy = self.current_df.copy()
        _metric_kwargs = metric_kwargs or {}

        # Row-wise calculation (no grouping)
        if group_by_cols is None:
            try:
                df_copy[metric_name] = df_copy.apply(
                    lambda row: metric(row, **_metric_kwargs), axis=1
                )
                self.metric_df = df_copy
                return df_copy
            except Exception as e:
                print(f"Error in row-wise calculation: {e}")
                return None

        # Grouped calculation
        try:
            # Validate grouping columns
            missing_cols = [col for col in group_by_cols if col not in df_copy.columns]
            if missing_cols:
                print(f"Error: Missing grouping columns: {missing_cols}")
                return None

            # Perform aggregation
            grouped = df_copy.groupby(group_by_cols, observed=True, sort=False)
            metric_func = lambda group: metric(group, **_metric_kwargs)
            result = grouped.apply(metric_func, include_groups=False)

            # Handle list-returning metrics
            if isinstance(result, pd.Series) and not result.empty:
                first_element = (
                    result.dropna().iloc[0] if not result.dropna().empty else None
                )
                if isinstance(first_element, (list, np.ndarray)):
                    result = result.explode()
                    result = pd.to_numeric(result, errors="coerce")

            # Ensure result is a Series
            if not isinstance(result, pd.Series):
                print(
                    f"Warning: Expected Series result, got {type(result)}. Attempting conversion."
                )
                if (
                    hasattr(result, "iloc")
                    and hasattr(result, "columns")
                    and len(result.columns) == 1
                ):
                    result = result.iloc[:, 0]
                else:
                    result = pd.Series(result)

            result.name = metric_name
            result_df = result.reset_index()
            self.metric_df = result_df
            return result_df

        except Exception as e:
            print(f"Error in grouped calculation: {e}")
            import traceback

            traceback.print_exc()
            return None

    def visualize(
        self,
        config: AnalysisConfig,
        data: pd.DataFrame = None,
    ) -> alt.Chart:
        """
        Visualizes the data using Altair charts.

        Args:
            config: Configuration for the visualization
            data: DataFrame to visualize (uses self.metric_df if None)
            show_chart: Whether to display the chart

        Returns:
            alt.Chart: The generated chart
        """
        try:
            # Use provided data or fall back to metric_df
            df_for_plot = data if data is not None else self.metric_df

            if df_for_plot is None or df_for_plot.empty:
                print("Error: No data available for visualization.")
                return (
                    alt.Chart()
                    .mark_text(text="No Data Available")
                    .properties(title="Error")
                )

            # --- Define encoding ---
            titles = config.titles or {}

            def get_title(category: Optional[str]) -> str:
                if not category:
                    return ""
                base_category = category.split(":")[0]
                return titles.get(
                    base_category, base_category.replace("_", " ").title()
                )

            # Basic encoding
            encoding = {}

            if config.x_category:
                encoding["x"] = alt.X(
                    config.x_category, title=get_title(config.x_category)
                )

            if config.y_category:
                encoding["y"] = alt.Y(
                    config.y_category, title=get_title(config.y_category)
                )

            # Add color encoding if specified
            if config.color_category:
                color_scale = alt.Scale()
                if config.color_scheme:
                    color_scale = alt.Scale(scheme=config.color_scheme)
                if config.color_domain and config.color_range:
                    color_scale = alt.Scale(
                        domain=config.color_domain, range=config.color_range
                    )
                elif config.color_domain:
                    color_scale = alt.Scale(domain=config.color_domain)
                elif config.color_range:
                    color_scale = alt.Scale(range=config.color_range)

                encoding["color"] = alt.Color(
                    config.color_category,
                    scale=color_scale,
                    title=get_title(config.color_category),
                    legend=config.color_legend or alt.Legend(**config.legend_config),
                )

            # Add additional encodings
            if config.opacity_category:
                encoding["opacity"] = alt.Opacity(
                    config.opacity_category,
                    title=get_title(config.opacity_category),
                    legend=config.opacity_legend,
                )

            if config.size_category:
                encoding["size"] = alt.Size(
                    config.size_category,
                    title=get_title(config.size_category),
                    legend=config.size_legend,
                )

            if config.shape_category:
                encoding["shape"] = alt.Shape(
                    config.shape_category,
                    title=get_title(config.shape_category),
                    legend=config.shape_legend,
                )

            if config.x_offset_category:
                encoding["xOffset"] = alt.XOffset(config.x_offset_category)

            if config.y_offset_category:
                encoding["yOffset"] = alt.YOffset(config.y_offset_category)

            # Add tooltips
            if config.tooltip_fields:
                encoding["tooltip"] = config.tooltip_fields

            # --- Create base chart ---
            plot_fn = config.plot_fn or alt.Chart.mark_point
            plot_fn_kwargs = config.plot_fn_kwargs or {}
            chart_properties = config.chart_properties or {}

            # Create chart with data, mark, encoding
            chart = plot_fn(alt.Chart(df_for_plot), **plot_fn_kwargs).encode(**encoding)

            # --- Handle layering ---
            if config.layer_category:
                base_layer_cat = config.layer_category.split(":")[0]
                if base_layer_cat in df_for_plot.columns:
                    chart = alt.layer(
                        *[
                            chart.transform_filter(f"datum.{base_layer_cat} == '{val}'")
                            for val in df_for_plot[base_layer_cat].dropna().unique()
                        ]
                    )

            # --- Handle faceting ---
            if config.facet_category:
                facet_field = config.facet_category.split(":")[0]
                if facet_field in df_for_plot.columns:
                    chart = chart.facet(
                        facet=alt.Facet(
                            config.facet_category,
                            title=get_title(config.facet_category),
                        ),
                        columns=config.facet_columns or 3,
                    )

            # --- Handle concatenation ---
            if config.h_concat_category:
                h_cat = config.h_concat_category.split(":")[0]
                if h_cat in df_for_plot.columns:
                    chart = alt.hconcat(
                        *[
                            chart.transform_filter(
                                f"datum.{h_cat} == '{val}'"
                            ).properties(
                                title=f"{get_title(config.h_concat_category)}: {val}"
                            )
                            for val in df_for_plot[h_cat].dropna().unique()
                        ],
                        resolve=alt.Resolve(
                            scale={
                                "y": "shared"
                                if config.shared_y_scale
                                else "independent"
                            }
                        ),
                    )

            if config.v_concat_category:
                v_cat = config.v_concat_category.split(":")[0]
                if v_cat in df_for_plot.columns:
                    chart = alt.vconcat(
                        *[
                            chart.transform_filter(
                                f"datum.{v_cat} == '{val}'"
                            ).properties(
                                title=f"{get_title(config.v_concat_category)}: {val}"
                            )
                            for val in df_for_plot[v_cat].dropna().unique()
                        ],
                        resolve=alt.Resolve(
                            scale={
                                "y": "shared"
                                if config.shared_y_scale
                                else "independent"
                            }
                        ),
                    )

            # --- Apply title and properties ---
            chart = chart.properties(title=config.fig_title, **chart_properties)

            # --- Add interactivity ---
            if config.interactive_chart:
                chart = chart.interactive()

             return chart

        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback

            traceback.print_exc()
            return (
                alt.Chart(pd.DataFrame({"error": [str(e)]}))
                .mark_text(text="error")
                .properties(title="Visualization Error")
            )

    # --- Helper methods (unchanged) ---
    def _apply_categorizers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal: Applies all stored categorizer functions."""
        temp_df = df.copy()

        for i, (output_columns, source_columns, categorizer) in enumerate(
            self._categorizers
        ):
            apply_axis = 0
            cols_to_apply = source_columns
            is_multi_source = isinstance(source_columns, tuple)

            # --- Input Validation ---
            if is_multi_source:
                cols_to_apply = list(source_columns)
                apply_axis = 1  # Apply row-wise for multiple columns
                if not all(c in temp_df.columns for c in cols_to_apply):
                    missing_cols = [
                        c for c in cols_to_apply if c not in temp_df.columns
                    ]
                    print(
                        f"Warning: Categorizer #{i + 1}: Source columns {missing_cols} not found. Skipping."
                    )
                    continue
            elif isinstance(source_columns, str):
                if source_columns not in temp_df.columns:
                    print(
                        f"Warning: Categorizer #{i + 1}: Source column '{source_columns}' not found. Skipping."
                    )
                    continue
            else:  # Should not happen based on add_categorizer validation
                print(
                    f"Warning: Categorizer #{i + 1}: Invalid source_columns type '{type(source_columns)}'. Skipping."
                )
                continue

            # --- Check for Output Column Conflicts ---
            output_cols_list = (
                [output_columns] if isinstance(output_columns, str) else output_columns
            )
            existing_cols = [c for c in output_cols_list if c in temp_df.columns]
            if existing_cols:
                print(
                    f"Warning: Categorizer #{i + 1}: Output column(s) {existing_cols} already exist. They will be overwritten."
                )
                # Optionally, could add a flag to raise an error instead

            # --- Apply Categorizer ---
            try:
                if apply_axis == 1:  # Multiple source columns
                    results = temp_df[cols_to_apply].apply(
                        lambda row: categorizer(*row), axis=apply_axis
                    )
                else:  # Single source column
                    results = temp_df[cols_to_apply].apply(categorizer)

                # --- Assign Results ---
                if results.empty:
                    # Handle empty results (e.g., assign NAs or skip)
                    for col_name in output_cols_list:
                        temp_df[col_name] = pd.NA
                    continue

                first_result = results.iloc[0]

                # Case 1: Output is a dictionary (multiple columns expected)
                if isinstance(first_result, dict):
                    if not isinstance(output_columns, list):
                        print(
                            f"Warning: Categorizer #{i + 1} returned a dictionary, but output_columns ('{output_columns}') is not a list. Skipping assignment."
                        )
                        continue
                    # Check if all results are dicts (or NaN)
                    if all(isinstance(item, dict) or pd.isna(item) for item in results):
                        try:
                            result_df = pd.DataFrame(
                                results.tolist(), index=temp_df.index
                            )
                            # Check if dict keys match expected output columns
                            if set(result_df.columns) != set(output_columns):
                                print(
                                    f"Warning: Categorizer #{i + 1} dictionary keys {list(result_df.columns)} do not match expected output_columns {output_columns}. Assigning based on expected names."
                                )
                                # Reindex result_df to match output_columns, filling missing with NaN
                                result_df = result_df.reindex(columns=output_columns)

                            # Assign columns
                            for col_name in output_columns:
                                temp_df[col_name] = result_df[col_name]

                        except Exception as e:
                            print(
                                f"Error processing dictionary results for Categorizer #{i + 1}: {e}"
                            )
                    else:
                        print(
                            f"Warning: Categorizer #{i + 1} did not consistently return dictionaries. Skipping assignment."
                        )

                # Case 2: Output is a single value (single column expected)
                else:
                    if not isinstance(output_columns, str):
                        print(
                            f"Warning: Categorizer #{i + 1} returned a single value, but output_columns ({output_columns}) is not a string. Skipping assignment."
                        )
                        continue
                    temp_df[output_columns] = results

            except Exception as e:
                print(
                    f"Error applying Categorizer #{i + 1} for source {source_columns} -> output {output_columns}: {e}"
                )

        return temp_df

    def _apply_renaming_and_sorting(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal: Applies stored renaming and sorting."""
        temp_df = df.copy()

        # Apply rename mappings first
        for column, mapping in self._rename_mappings.items():
            if column in temp_df.columns:
                temp_df[column] = temp_df[column].map(mapping).fillna(temp_df[column])

        # Apply custom sort order (which filters and sets categorical type)
        sort_columns_present = []
        for column, order in self._sort_orders.items():
            if column in temp_df.columns:
                # Filter based on the provided order
                temp_df = temp_df[temp_df[column].isin(order)]
                if not temp_df.empty:
                    # Set the categorical type with the specified order
                    temp_df[column] = pd.Categorical(
                        temp_df[column], categories=order, ordered=True
                    )
                    sort_columns_present.append(column)
            else:
                print(f"Warning: Column '{column}' specified in sort_order not found in DataFrame.")

        # Sort the dataframe based on all columns with custom sort order
        if sort_columns_present and not temp_df.empty:
            temp_df = temp_df.sort_values(sort_columns_present, na_position="last")
        
        # Apply callable sort functions (if any exist)
        if hasattr(self, '_sort_functions') and self._sort_functions:
            for sort_func in self._sort_functions:
                try:
                    temp_df = sort_func(temp_df)
                except Exception as e:
                    print(f"Warning: Error applying sort function: {e}")
                
        return temp_df

    def filter_and_sort(self, spec: Union[Dict[str, List], Callable[[pd.DataFrame], pd.DataFrame]]):
        """
        Adds both filtering and sorting based on the same specification.
        
        This is a convenience method that combines add_filter() and set_sort_order()
        when you want to filter and sort on the same values.
        
        Args:
            spec: Either:
                - A dictionary mapping columns to allowed values: {"column": [val1, val2]}
                  Values will be filtered to only these values and sorted in the given order
                - A callable that takes a DataFrame and returns a filtered and sorted DataFrame
        
        Returns:
            self: For method chaining
        """
        if callable(spec):
            # Add the callable as both a filter and a sort function
            self.add_filter(spec)
            self.set_sort_order(spec)
        elif isinstance(spec, dict):
            # Add a dictionary-based filter
            self.add_filter(spec)
            # Use the same dictionary for sort order
            self.set_sort_order(spec)
        else:
            raise ValueError("spec must be either a dictionary or a callable function.")
        self.process_dataframe()  # Automatically process after filter and sort
        return self


if __name__ == "__main__":
    from metrics import pairwise_bce_of_group

    logprob_data_path = "data/logprobs.csv"
    analyzer = LogprobAnalyzer(logprob_data_path)

    # Rename mapping for nicer labels
    analyzer.add_rename_mapping(
        "model_name",
        {
            "openai-community/gpt2-medium": "GPT-2-M",
            "meta-llama/Llama-3.2-1B": "Llama3.2-1B",
        },
    )

    analyzer.filter_and_sort({"model_name": ["GPT-2-M", "Llama3.2-1B"]})

    # Set sort order using the *new* names
    analyzer.set_sort_order("model_name", ["GPT-2-M", "Llama3.2-1B"])

    # Make sure to process the dataframe and calculate metrics before visualization
    analyzer.process_dataframe()
    
    # Calculate metrics for variance method
    analyzer.calculate_metric(
        metric=pairwise_error_of_group,
        metric_name="pairwise_bce_mse",
        group_by_cols=["model_name", "temperature", "class_type"],
        metric_kwargs={"value_col": "_bce_sum", "square": False}
    )

    # --- Plot 1: Variance BCE ---
    config_var = AnalysisConfig(
        plot_fn=alt.Chart.mark_line,
        fig_title="BCE (Variance method) by Model and Temperature",
        x_category="temperature:Q",
        y_category="variance(single_evidence_estimate):Q",
        facet_category="model_name:N",
        # facet_columns=2, # Let the code calculate default columns
        tooltip_fields=[
            alt.Tooltip("model_name:N", title="Model"),
            alt.Tooltip("temperature:Q", title="Temp"),
            alt.Tooltip(
                "variance(single_evidence_estimate):Q", title="BCE", format=".3f"
            ),
            alt.Tooltip("count():Q", title="Count", format="d"),
        ],
        titles={
            "variance(single_evidence_estimate)": "BCE (Variance method)",
            "model_name": "Language Model",
            "temperature": "Temperature",
        },
        interactive_chart=False,
        legend_config={"orient": "top"},
    )

    chart_var = analyzer.visualize(config=config_var, data=None, show_chart=True)
    chart_var.show()  # Display the chart

    # Calculate metrics for pairwise MSE method
    analyzer.calculate_metric(
        metric=pairwise_mse_of_group,
        metric_name="pairwise_bce_mse",
        group_by_cols=["model_name", "temperature"],
        metric_kwargs={}
    )

    # --- Plot 2: Pairwise MSE BCE ---
    config_mse = AnalysisConfig(
        plot_fn=alt.Chart.mark_boxplot,
        fig_title="BCE (Pairwise MSE method) by Model and Temperature",
        x_category="temperature:Q",
        y_category="pairwise_bce_mse:Q",
        facet_category="model_name:N",
        # facet_columns=2, # Let the code calculate default columns
        tooltip_fields=[  # Using explicit tooltips
            alt.Tooltip("model_name:N", title="Model"),
            alt.Tooltip("temperature:Q", title="Temp"),
            alt.Tooltip("mean(pairwise_bce_mse):Q", title="Mean BCE", format=".3f"),
            alt.Tooltip("count():Q", title="Count", format="d"),
        ],
        titles={
            "pairwise_bce_mse": "BCE (Pairwise MSE method)",
            "model_name": "Language Model",
            "temperature": "Temperature",
        },
        interactive_chart=False,
        legend_config={"orient": "top"},
    )

    chart_mse = analyzer.visualize(config=config_mse, data=None, show_chart=True)
    chart_mse.show()  # Display the chart
