import pandas as pd
import altair as alt
from altair import Undefined
from typing import List, Optional, Callable, Dict, Any, Union, Tuple, Literal
import math
from dataclasses import dataclass, field
import numpy as np


@dataclass
class AnalysisConfig:
    """Configuration for the LogprobAnalyzer visualization."""
    plot_fn: Callable[..., alt.Chart] = alt.Chart.mark_point # Default to point plot
    fig_title: Optional[str] = None
    plot_fn_kwargs: Dict[str, Any] = field(default_factory=dict)
    chart_properties: Dict[str, Any] = field(default_factory=dict)
    x_category: Optional[str] = None # Column for X-axis
    y_category: Optional[str] = None # Column for Y-axis (should match metric_name used in Analyzer)
    x_offset_category: Optional[str] = None
    y_offset_category: Optional[str] = None
    color_category: Optional[str] = None
    color_scheme: Optional[str] = None
    color_domain: Optional[List[str]] = None
    color_range: Optional[List[str]] = None
    color_legend: Optional[alt.Legend] = None
    opacity_category: Optional[str] = None
    opacity_legend: Optional[alt.Legend] = None
    size_category: Optional[str] = None # Added for scatter plots etc.
    size_legend: Optional[alt.Legend] = None
    shape_category: Optional[str] = None # Added for different shapes
    shape_legend: Optional[alt.Legend] = None
    layer_category: Optional[str] = None
    facet_category: Optional[str] = None
    facet_columns: Optional[int] = None
    h_concat_category: Optional[str] = None
    v_concat_category: Optional[str] = None
    shared_y_scale: bool = False
    tooltip_fields: Optional[List[alt.Tooltip]] = None
    titles: Optional[Dict[str, str]] = field(default_factory=dict) # Titles for axes/legends
    legend_config: Dict[str, Any] = field(default_factory=lambda: {
        "orient": "bottom",
        "columns": 3,
        "titleAlign": "center",
        "labelLimit": 1000
    })
    interactive_chart: bool = False # Enable zooming and panning


# Assuming AnalysisConfig is defined in analysis_config.py or above
# from analysis_config import AnalysisConfig

class LogprobAnalyzer:
    """
    Analyzes and visualizes log probability data interactively using Altair charts.

    Initialize with a DataFrame or CSV path. Use methods like `add_filter`,
    `add_categorizer`, `add_rename_mapping`, `set_sort_order` to define
    processing steps. Finally, call `visualize` with a metric specification and
    configuration to generate the plot.
    """

    def __init__(self, source: Union[pd.DataFrame, str]):
        """
        Initialize the LogprobAnalyzer.

        Args:
            source (Union[pd.DataFrame, str]): The input data, either as a
                pandas DataFrame or a file path to a CSV.
        """
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
        self._filters: List[Callable[[pd.DataFrame], pd.DataFrame]] = []
        self._categorizers: List[Tuple[Union[str, List[str]], Union[str, Tuple[str, ...]], Callable]] = []
        self._rename_mappings: Dict[str, Dict[str, str]] = {}
        self._sort_orders: Dict[str, List[str]] = {}

        # Store the currently processed state of the DataFrame
        self.current_df = self.original_df.copy()

    def add_filter(self, filter_fn: Callable[[pd.DataFrame], pd.DataFrame]):
        """
        Adds a filter function to the processing pipeline.

        Args:
            filter_fn: A function that takes a DataFrame and returns a
                       filtered DataFrame.
        """
        if not callable(filter_fn):
            raise ValueError("filter_fn must be a callable function.")
        self._filters.append(filter_fn)
        # Re-process the dataframe immediately to reflect the change
        self.process_dataframe()
        return self # Allow chaining

    def add_categorizer(self,
                        output_columns: Union[str, List[str]],
                        categorizer: Callable,
                        source_columns: Union[str, Tuple[str, ...]]):
        """
        Adds a categorizer function to create new columns with explicit names.

        Args:
            output_columns: The name (str) or list of names (List[str]) for
                            the new column(s) to be created.
            categorizer: A function that takes the value(s) from source_columns
                         (as separate arguments if multiple columns) and returns
                         either a single value (if output_columns is str) or a
                         dictionary (if output_columns is List[str], keys must
                         match the list).
            source_columns: The source column name (str) or tuple/list of names
                            to pass to the categorizer.
        """
        if not callable(categorizer):
             raise ValueError("categorizer must be a callable function.")
        if isinstance(source_columns, list): # Accept list as well as tuple
            source_columns = tuple(source_columns)
        if not isinstance(output_columns, (str, list)):
             raise ValueError("output_columns must be a string or a list of strings.")
        if isinstance(output_columns, list) and not all(isinstance(c, str) for c in output_columns):
             raise ValueError("If output_columns is a list, all elements must be strings.")

        self._categorizers.append((output_columns, source_columns, categorizer))
        self.process_dataframe()
        return self

    def add_rename_mapping(self, column: str, mapping: Dict[str, str]):
        """
        Adds or updates rename mappings for values within a specific column.

        Args:
            column: The name of the column whose values should be renamed.
            mapping: A dictionary mapping {old_value: new_value}.
        """
        if column not in self._rename_mappings:
            self._rename_mappings[column] = {}
        self._rename_mappings[column].update(mapping)
        self.process_dataframe()
        return self

    def set_sort_order(self, column: str, order: List[str]):
        """
        Sets a custom sort order for a categorical column and filters
        the DataFrame to include only values present in this order.
        The order should use the *final* (potentially renamed) values.

        Args:
            column: The name of the column to sort.
            order: A list of category values in the desired sort order.
                   Only rows with values in this list will be kept.
        """
        self._sort_orders[column] = order
        self.process_dataframe()
        return self

    def reset_processing(self):
        """Resets all added filters, categorizers, renames, and sorts."""
        self._filters = []
        self._categorizers = [] # Reset categorizers list
        self._rename_mappings = {}
        self._sort_orders = {}
        self.current_df = self.original_df.copy()
        print("Processing steps reset. DataFrame reverted to original.")
        return self

    def process_dataframe(self) -> pd.DataFrame:
        """
        Applies all stored filters, categorizers, renames, and sorts
        sequentially starting from the original DataFrame.

        Updates `self.current_df` with the result. Usually called internally
        after adding a processing step, but can be called manually.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        df = self.original_df.copy() # Always start fresh

        # Apply filters
        for i, f in enumerate(self._filters):
            try:
                df = f(df)
            except Exception as e:
                print(f"Warning: Error applying filter #{i+1} ({getattr(f, '__name__', 'anonymous')}): {e}. Skipping filter.")

        # Apply categorizers
        df = self._apply_categorizers(df)

        # Apply renaming and sorting
        df = self._apply_renaming_and_sorting(df)

        self.current_df = df
        return self.current_df

    # --- Internal Helper Methods ---

    def _apply_categorizers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal: Applies all stored categorizer functions using explicit output names."""
        temp_df = df.copy()

        for i, (output_columns, source_columns, categorizer) in enumerate(self._categorizers):
            apply_axis = 0
            cols_to_apply = source_columns
            is_multi_source = isinstance(source_columns, tuple)

            # --- Input Validation ---
            if is_multi_source:
                cols_to_apply = list(source_columns)
                apply_axis = 1 # Apply row-wise for multiple columns
                if not all(c in temp_df.columns for c in cols_to_apply):
                    missing_cols = [c for c in cols_to_apply if c not in temp_df.columns]
                    print(f"Warning: Categorizer #{i+1}: Source columns {missing_cols} not found. Skipping.")
                    continue
            elif isinstance(source_columns, str):
                if source_columns not in temp_df.columns:
                    print(f"Warning: Categorizer #{i+1}: Source column '{source_columns}' not found. Skipping.")
                    continue
            else: # Should not happen based on add_categorizer validation
                 print(f"Warning: Categorizer #{i+1}: Invalid source_columns type '{type(source_columns)}'. Skipping.")
                 continue

            # --- Check for Output Column Conflicts ---
            output_cols_list = [output_columns] if isinstance(output_columns, str) else output_columns
            existing_cols = [c for c in output_cols_list if c in temp_df.columns]
            if existing_cols:
                print(f"Warning: Categorizer #{i+1}: Output column(s) {existing_cols} already exist. They will be overwritten.")
                # Optionally, could add a flag to raise an error instead

            # --- Apply Categorizer ---
            try:
                if apply_axis == 1: # Multiple source columns
                    results = temp_df[cols_to_apply].apply(
                        lambda row: categorizer(*row), axis=apply_axis
                    )
                else: # Single source column
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
                        print(f"Warning: Categorizer #{i+1} returned a dictionary, but output_columns ('{output_columns}') is not a list. Skipping assignment.")
                        continue
                    # Check if all results are dicts (or NaN)
                    if all(isinstance(item, dict) or pd.isna(item) for item in results):
                        try:
                            result_df = pd.DataFrame(results.tolist(), index=temp_df.index)
                            # Check if dict keys match expected output columns
                            if set(result_df.columns) != set(output_columns):
                                print(f"Warning: Categorizer #{i+1} dictionary keys {list(result_df.columns)} do not match expected output_columns {output_columns}. Assigning based on expected names.")
                                # Reindex result_df to match output_columns, filling missing with NaN
                                result_df = result_df.reindex(columns=output_columns)

                            # Assign columns
                            for col_name in output_columns:
                                temp_df[col_name] = result_df[col_name]

                        except Exception as e:
                             print(f"Error processing dictionary results for Categorizer #{i+1}: {e}")
                    else:
                        print(f"Warning: Categorizer #{i+1} did not consistently return dictionaries. Skipping assignment.")

                # Case 2: Output is a single value (single column expected)
                else:
                    if not isinstance(output_columns, str):
                        print(f"Warning: Categorizer #{i+1} returned a single value, but output_columns ({output_columns}) is not a string. Skipping assignment.")
                        continue
                    temp_df[output_columns] = results

            except Exception as e:
                print(f"Error applying Categorizer #{i+1} for source {source_columns} -> output {output_columns}: {e}")

        return temp_df

    def _apply_renaming_and_sorting(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal: Applies stored renaming and sorting."""
        temp_df = df.copy()

        # Apply rename mappings first
        for column, mapping in self._rename_mappings.items():
            if column in temp_df.columns:
                # Ensure mapping keys are handled correctly (e.g., map only existing values)
                temp_df[column] = temp_df[column].map(mapping).fillna(temp_df[column])
            # else: # Don't warn here, column might be created by categorizer later
            #     print(f"Warning: Column '{column}' specified in rename_mappings not found.")


        # Apply custom sort order (which filters and sets categorical type)
        sort_columns_present = []
        for column, order in self._sort_orders.items():
            if column in temp_df.columns:
                 # Filter based on the provided order (should contain final/renamed values)
                temp_df = temp_df[temp_df[column].isin(order)]
                if not temp_df.empty:
                    # Set the categorical type with the specified order
                    temp_df[column] = pd.Categorical(
                        temp_df[column], categories=order, ordered=True
                    )
                    sort_columns_present.append(column)
                # else: # If filtering made df empty, no need to set category or sort
                #     print(f"Warning: Filtering by sort order for '{column}' resulted in empty DataFrame.")

            else:
                print(f"Warning: Column '{column}' specified in sort_order not found in DataFrame.")

        # Sort the dataframe based on all columns with custom sort order
        if sort_columns_present and not temp_df.empty:
            # Sort, handling potential NaNs introduced by categorization/filtering
            temp_df = temp_df.sort_values(sort_columns_present, na_position='last')


        return temp_df

    def _prepare_plot_data(self,
                           metric: Callable,
                           metric_name: str,
                           aggregate: bool,
                           group_by_cols: Optional[List[str]],
                           metric_kwargs: Dict,
                           config: AnalysisConfig # Added config for potential use
                          ) -> tuple[Optional[pd.DataFrame], Optional[List[str]]]:
        """
        Handles validation and metric calculation (row-wise or aggregate).
        Returns the DataFrame ready for plotting and the grouping columns used.
        Returns (None, None) on failure.
        """
        if self.current_df is None:
            print("Error: Data not loaded.")
            return None, None

        df_copy = self.current_df.copy()
        final_group_by_cols = []

        # --- Validate Metric Name ---
        if aggregate and metric_name in df_copy.columns:
            print(f"Warning: Aggregate metric name '{metric_name}' already exists as a column. It might be overwritten if grouping matches.")
        elif not aggregate and metric_name in df_copy.columns:
             print(f"Warning: Row-wise metric name '{metric_name}' already exists. It will be overwritten.")


        # --- Handle Aggregation vs. Row-wise ---
        if aggregate:
            # Determine grouping columns
            group_by_cols_set = set(group_by_cols or [])
            # Add categories used in plot config to group_by if they exist and are not already included
            # Define relevant config attributes that imply grouping
            grouping_relevant_attrs = [
                'x_category', 'y_category', 'color_category', 'row_category',
                'column_category', 'facet_category', 'layer_category',
                'h_concat_category', 'v_concat_category', 'x_offset_category',
                'y_offset_category', 'opacity_category', 'shape_category', 'size_category'
            ]
            for attr in grouping_relevant_attrs:
                 cat = getattr(config, attr, None)
                 if cat:
                     # Add the base column name (without type specifier)
                     group_by_cols_set.add(cat.split(':')[0])

            final_group_by_cols = sorted(list(group_by_cols_set - {metric_name})) # Exclude metric name itself if used as category

            if not final_group_by_cols:
                print("Error: Aggregation requested but no grouping columns specified or derived from config.")
                return None, None

            missing_cols = [col for col in final_group_by_cols if col not in df_copy.columns]
            if missing_cols:
                print(f"Error: Grouping columns {missing_cols} not found in DataFrame.")
                return None, None

            print(f"Calculating aggregate metric '{metric_name}' using '{metric.__name__}' grouped by {final_group_by_cols}...")
            # Use observed=True for categorical grouping, sort=False to respect existing sort order
            grouped = df_copy.groupby(final_group_by_cols, observed=True, sort=False)

            # Use apply for flexibility, passing metric_kwargs if needed
            metric_func_to_apply = lambda group: metric(group, **metric_kwargs) if metric_kwargs else metric(group)

            try:
                # Pass include_groups=False to avoid the deprecation warning and potential issues
                agg_result = grouped.apply(metric_func_to_apply, include_groups=False)

                # --- Add Explode step for list-returning metrics like pairwise MSE ---
                # Check if the result is a Series and its elements are lists/arrays
                if isinstance(agg_result, pd.Series) and not agg_result.empty:
                    first_valid_element = agg_result.dropna().iloc[0] if not agg_result.dropna().empty else None
                    if isinstance(first_valid_element, (list, np.ndarray)):
                        print(f"Detected list output for metric '{metric_name}'. Exploding results for plotting.")
                        agg_result = agg_result.explode().dropna()
                        agg_result = pd.to_numeric(agg_result, errors='coerce').dropna()
                # --- End Explode step ---

            except Exception as e:
                 print(f"Error during aggregation calculation: {e}")
                 # import traceback
                 # traceback.print_exc() # Uncomment for detailed traceback
                 return None, None


            if not isinstance(agg_result, pd.Series):
                 print(f"Warning: Aggregation metric did not return a Pandas Series (returned {type(agg_result)}). Attempting conversion.")
                 try:
                     if isinstance(agg_result, pd.DataFrame) and len(agg_result.columns) == 1:
                         agg_result = agg_result.iloc[:, 0]
                     elif not hasattr(agg_result, 'index'):
                          print("Aggregation returned a scalar value. Ensure this is intended.")
                          # Attempt to reconstruct a Series - might fail for complex cases
                          # Ensure index matches the group keys
                          if grouped.ngroups == 1:
                              agg_result = pd.Series([agg_result], index=grouped.groups.keys())
                          else:
                              # This case is ambiguous without knowing the group structure
                              print("Error: Cannot reliably convert scalar result to Series for multiple groups.")
                              return None, None
                     else: # Attempt direct conversion
                         agg_result = pd.Series(agg_result)

                 except Exception as conv_e:
                     print(f"Error: Could not convert aggregation result to Series: {conv_e}")
                     return None, None


            agg_result.name = metric_name
            df_for_plot = agg_result.reset_index()
            print(f"Aggregation complete. Plotting {len(df_for_plot)} points.")

        else:
            # Row-wise calculation
            print(f"Calculating row-wise metric '{metric_name}' using '{metric.__name__}'...")
            try:
                # Apply row-wise, passing metric_kwargs if they exist
                df_copy[metric_name] = df_copy.apply(
                    lambda row: metric(row, **metric_kwargs) if metric_kwargs else metric(row),
                    axis=1
                )
                df_for_plot = df_copy
                print("Row-wise calculation complete.")
            except Exception as e:
                 print(f"Error during row-wise calculation for '{metric_name}': {e}")
                 # import traceback
                 # traceback.print_exc() # Uncomment for detailed traceback
                 return None, None


        # --- Final Validation ---
        required_plot_cols = set()
        # Check columns used directly in encodings/facets etc.
        check_attrs = [
            'x_category', 'y_category', 'color_category', 'facet_category',
            'row_category', 'column_category', 'opacity_category', 'size_category',
            'shape_category', 'layer_category', 'h_concat_category', 'v_concat_category',
            'x_offset_category', 'y_offset_category'
        ]
        for attr in check_attrs:
             cat = getattr(config, attr, None)
             if cat:
                 required_plot_cols.add(cat.split(':')[0])

        # The metric name itself is always required if it's not already a category
        if metric_name not in required_plot_cols:
             required_plot_cols.add(metric_name)


        missing_plot_cols = [col for col in required_plot_cols if col not in df_for_plot.columns]
        if missing_plot_cols:
             print(f"Error: Columns required for plotting are missing from the prepared data: {missing_plot_cols}")
             print(f"Available columns: {df_for_plot.columns.tolist()}")
             return None, None


        return df_for_plot, final_group_by_cols # Return group cols used

    def _get_title(self, category: Optional[str], titles_dict: Optional[Dict[str, str]]) -> str:
        """Gets a title for a category, using defaults if necessary."""
        # Ensure titles_dict is usable, default to empty dict if None or missing
        safe_titles_dict = titles_dict if isinstance(titles_dict, dict) else {}

        if not category:
            return "" # Return empty string if category is None or empty

        # Remove type encoding like ':N', ':Q', ':O', ':T' for lookup and title generation
        base_category = category.split(':')[0]

        # Return custom title if found, otherwise format the base category name
        return safe_titles_dict.get(base_category, base_category.replace('_', ' ').title())

    def _build_encodings(self, config: AnalysisConfig, df_for_plot: pd.DataFrame, metric_name: str, aggregate: bool) -> Dict[str, Any]:
        """Builds the Altair encoding dictionary based on the config."""
        encoding = {}

        # Map channel names to Altair encoding classes and config attributes
        channel_map = {
            'x': (alt.X, 'x_category'),
            'y': (alt.Y, 'y_category'),
            'color': (alt.Color, 'color_category'),
            'opacity': (alt.Opacity, 'opacity_category'),
            'size': (alt.Size, 'size_category'),
            'shape': (alt.Shape, 'shape_category'),
            'row': (alt.Row, 'row_category'), # Added row/column for completeness
            'column': (alt.Column, 'column_category'),
            'xOffset': (alt.XOffset, 'x_offset_category'),
            'yOffset': (alt.YOffset, 'y_offset_category'),
            # Layer is handled structurally, not as a direct encoding channel
        }

        # Helper to create encoding channel if category is defined
        def add_encoding(channel_name: str, encoding_class: type, category_attr: str):
            category = getattr(config, category_attr, None)
            if category:
                # Apply title using the specific channel class
                encoding[channel_name] = encoding_class(category, title=self._get_title(category, config.titles))
            # Add specific legend handling if needed (e.g., color, opacity, size, shape)
            legend_attr = f"{channel_name}_legend"
            legend_obj = getattr(config, legend_attr, None)
            if category and legend_obj is not None and hasattr(encoding[channel_name], 'legend'):
                 encoding[channel_name].legend = legend_obj
            elif category and channel_name == 'color' and legend_obj is None and getattr(config, 'legend_config', None):
                 # Apply general legend_config to color if no specific color_legend is set
                 encoding[channel_name].legend = alt.Legend(**config.legend_config)

        # Add encodings based on the map
        for ch_name, (ch_class, ch_attr) in channel_map.items():
            add_encoding(ch_name, ch_class, ch_attr)

        # --- Tooltip Encoding ---
        tooltip = []
        tooltip_items_to_process: Optional[List[Union[str, alt.SchemaBase]]] = getattr(config, 'tooltip_fields', None)

        if tooltip_items_to_process is None:
            # Simplified Default tooltips: include essential columns + metric + count (if agg)
            default_tooltip_cols_spec = []
            processed_bases = set() # Track base column names to avoid duplicates

            # Add core encodings if they exist
            for attr in ['x_category', 'y_category', 'color_category', 'row_category', 'column_category', 'facet_category']:
                cat = getattr(config, attr, None)
                if cat:
                    base_cat = cat.split(':')[0]
                    if base_cat not in processed_bases:
                        default_tooltip_cols_spec.append(cat)
                        processed_bases.add(base_cat)

            # Add the metric name itself if it's a column and not already added
            metric_base = metric_name.split(':')[0] # Use base name for check
            if metric_name in df_for_plot.columns and metric_base not in processed_bases:
                # Add with type if possible, default to quantitative ':Q' if no type in name
                metric_spec = metric_name if ':' in metric_name else f"{metric_name}:Q"
                default_tooltip_cols_spec.append(metric_spec)
                processed_bases.add(metric_base)

            # Add count if it's an aggregated plot
            if aggregate:
                # Check if 'count' column exists from aggregation (unlikely with current apply)
                # Prefer Altair's count() aggregation for tooltips
                if 'count()' not in processed_bases:
                    tooltip.append(alt.Tooltip("count()", title="Count")) # Use alt.Tooltip object

            # Use the generated defaults (strings only for now)
            tooltip_items_to_process = default_tooltip_cols_spec
            # print(f"Debug: Using simplified default tooltips: {tooltip_items_to_process}") # Optional debug print

        # Process the final list of tooltip items (either user-provided or default)
        if isinstance(tooltip_items_to_process, list):
            processed_tooltip_bases = set() # Keep track to avoid duplicate base columns in warnings/tooltips
            for field_spec in tooltip_items_to_process:
                field_name_to_check = None
                field_to_add = field_spec # Default to adding the original item

                try:
                    if isinstance(field_spec, str):
                        field_name_to_check = field_spec.split(':')[0]
                    elif hasattr(field_spec, 'field') and isinstance(getattr(field_spec, 'field', None), str):
                        field_name_to_check = field_spec.field.split(':')[0]
                    elif hasattr(field_spec, 'shorthand') and isinstance(getattr(field_spec, 'shorthand', None), str):
                         field_name_to_check = field_spec.shorthand.split(':')[0]
                    else:
                         print(f"Warning: Could not extract field name from tooltip item: {field_spec}. Adding it without validation.")

                    if field_name_to_check:
                        is_aggregation = field_name_to_check.endswith('()')
                        base_field_name = field_name_to_check[:-2] if is_aggregation else field_name_to_check

                        if base_field_name not in processed_tooltip_bases:
                             processed_tooltip_bases.add(base_field_name)
                             if not is_aggregation and base_field_name not in df_for_plot.columns:
                                 print(f"Warning: Tooltip field '{base_field_name}' not found in DataFrame columns: {df_for_plot.columns.tolist()}")
                             tooltip.append(field_to_add) # Add the original specifier

                except Exception as e:
                    print(f"Error processing tooltip item '{field_spec}': {e}. Skipping this item.")

        if tooltip: # Only add tooltip encoding if there's something to show
             encoding['tooltip'] = tooltip

        # Apply specific color scale properties if defined
        if 'color' in encoding:
            if config.color_scheme:
                encoding['color'].scale = alt.Scale(scheme=config.color_scheme)
            if config.color_domain and config.color_range:
                 encoding['color'].scale = alt.Scale(domain=config.color_domain, range=config.color_range)
            elif config.color_domain:
                 encoding['color'].scale = alt.Scale(domain=config.color_domain)
            elif config.color_range:
                 encoding['color'].scale = alt.Scale(range=config.color_range)

        return encoding

    def _build_chart_structure(self,
                               base_chart: alt.Chart, # Takes the chart after mark and encode
                               config: AnalysisConfig,
                               df_for_plot: pd.DataFrame,
                               encoding: Dict) -> alt.Chart:
         """Applies layering, faceting, and concatenation."""
         chart = base_chart
         titles = config.titles or {}

         # --- Layering ---
         if config.layer_category:
             base_layer_cat = config.layer_category.split(':')[0]
             if base_layer_cat not in df_for_plot.columns:
                  print(f"Warning: layer_category '{base_layer_cat}' not found. Ignoring layering.")
             else:
                 try:
                     unique_layers = df_for_plot[base_layer_cat].dropna().unique()
                     # Resolve scales independently based on what was encoded
                     resolve_args = {
                         scale: 'independent' for scale in ['color', 'opacity', 'size', 'shape'] if scale in encoding
                     }
                     chart = alt.layer(
                         *[
                             # Apply filter to the base chart *before* layering
                             base_chart.transform_filter(alt.datum[base_layer_cat] == val)
                             for val in unique_layers
                         ]
                     ).resolve_scale(**resolve_args)
                 except Exception as e:
                     print(f"Error during layering on '{base_layer_cat}': {e}")
                     chart = base_chart # Revert to non-layered on error

         # --- Faceting ---
         facet_category = getattr(config, 'facet_category', None)
         facet_columns = getattr(config, 'facet_columns', None) # Check for explicit columns in config

         if facet_category:
             facet_field_name = facet_category.split(':')[0]
             if facet_field_name not in df_for_plot.columns:
                  print(f"Warning: Facet category '{facet_field_name}' not found in data. Skipping faceting.")
                  facet_category = None # Disable faceting
             else:
                 # Determine the number of columns for faceting if not specified
                 if facet_columns is None:
                     try:
                         num_unique = df_for_plot[facet_field_name].nunique()
                         # Set a reasonable default, e.g., max 3 columns or sqrt
                         facet_columns = min(num_unique, 3) # Default to max 3 columns
                         if facet_columns == 0:
                             print(f"Warning: Facet category '{facet_field_name}' has 0 unique values. Skipping faceting.")
                             facet_category = None # Disable faceting
                         elif num_unique > 9: # Warn if potentially too many facets per row
                             print(f"Warning: Facet category '{facet_field_name}' has {num_unique} unique values. Consider setting 'facet_columns' in AnalysisConfig for better layout.")
                     except Exception as e:
                          print(f"Warning: Could not determine number of columns for facet '{facet_category}'. Skipping faceting. Error: {e}")
                          facet_category = None # Disable faceting

                 # Validate facet_columns before applying
                 if facet_category and (not isinstance(facet_columns, int) or facet_columns <= 0):
                      print(f"Warning: Invalid number of columns ({facet_columns}) calculated or provided for faceting. Skipping faceting.")
                      facet_category = None # Disable faceting

             # Apply faceting only if category and columns are valid
             if facet_category:
                  print(f"Applying faceting on '{facet_category}' with {facet_columns} columns.")
                  try:
                      # Apply facet to the potentially layered chart
                      chart = chart.facet(
                          facet=alt.Facet(facet_category, title=self._get_title(facet_category, titles)),
                          columns=facet_columns # Explicitly provide columns
                      )
                  except Exception as facet_e:
                       print(f"Error applying faceting: {facet_e}. Proceeding without faceting.")
                       # Chart remains as it was before attempting facet

         # --- Concatenation ---
         # Note: Concatenation operates on the potentially layered/faceted chart
         charts_to_concat = [chart] # Start with the current chart state
         resolve_y = alt.Resolve(scale={"y": "shared" if config.shared_y_scale else "independent"})

         if config.h_concat_category:
             base_hconcat_cat = config.h_concat_category.split(':')[0]
             if base_hconcat_cat not in df_for_plot.columns:
                  print(f"Warning: h_concat_category '{base_hconcat_cat}' not found. Ignoring hconcat.")
             else:
                 try:
                     unique_hconcat = df_for_plot[base_hconcat_cat].dropna().unique()
                     processed_charts = []
                     for c in charts_to_concat: # Apply hconcat to each chart we have so far
                          hconcat_list = []
                          for val in unique_hconcat:
                               title_str = f"{self._get_title(config.h_concat_category, titles)}: {val}" if len(unique_hconcat) > 1 else Undefined
                               filtered_chart = c.transform_filter(
                                    alt.datum[base_hconcat_cat] == val
                               ).properties(title=title_str)
                               hconcat_list.append(filtered_chart)
                          processed_charts.append(alt.hconcat(*hconcat_list, resolve=resolve_y))
                     charts_to_concat = processed_charts # Update the list of charts
                 except Exception as e:
                      print(f"Error during horizontal concatenation on '{base_hconcat_cat}': {e}")

         if config.v_concat_category:
             base_vconcat_cat = config.v_concat_category.split(':')[0]
             if base_vconcat_cat not in df_for_plot.columns:
                  print(f"Warning: v_concat_category '{base_vconcat_cat}' not found. Ignoring vconcat.")
             else:
                 try:
                     unique_vconcat = df_for_plot[base_vconcat_cat].dropna().unique()
                     processed_charts = []
                     for c in charts_to_concat: # Apply vconcat potentially after hconcat
                          vconcat_list = []
                          for val in unique_vconcat:
                               title_str = f"{self._get_title(config.v_concat_category, titles)}: {val}" if len(unique_vconcat) > 1 else Undefined
                               filtered_chart = c.transform_filter(
                                    alt.datum[base_vconcat_cat] == val
                               ).properties(title=title_str)
                               vconcat_list.append(filtered_chart)
                          processed_charts.append(alt.vconcat(*vconcat_list, resolve=resolve_y))
                     charts_to_concat = processed_charts # Update the list of charts
                 except Exception as e:
                      print(f"Error during vertical concatenation on '{base_vconcat_cat}': {e}")

         # --- Final Chart Assembly ---
         if len(charts_to_concat) == 1:
             final_chart = charts_to_concat[0]
         elif len(charts_to_concat) > 1:
             # This case implies multiple independent charts resulted from concat steps
             # Combine them vertically by default
             print("Warning: Multiple independent charts resulted from concatenation steps, combining vertically.")
             final_chart = alt.vconcat(*charts_to_concat)
         else: # Should not happen if base_chart was valid
             raise RuntimeError("Chart generation failed unexpectedly (no charts produced after structuring).")

         return final_chart


    def visualize(self,
                  config: AnalysisConfig,
                  metric: Callable,
                  metric_name: str,
                  aggregate: bool,
                  metric_kwargs: Optional[Dict] = None, # Use kwargs for metric args
                  group_by_cols: Optional[List[str]] = None,
                  show_chart: bool = False # Control display
                 ) -> alt.Chart:
        """
        Core visualization pipeline: Calculates metric, builds chart config, generates chart.
        """
        # Ensure metric_kwargs is a dict for internal use
        _metric_kwargs = metric_kwargs or {}

        try:
            # --- Data Preparation ---
            df_for_plot, _ = self._prepare_plot_data(
                metric=metric,
                metric_name=metric_name,
                aggregate=aggregate,
                group_by_cols=group_by_cols,
                metric_kwargs=_metric_kwargs, # Pass the kwargs here
                config=config
            )
            if df_for_plot is None:
                 print("Error: Data preparation failed. Cannot generate plot.")
                 return alt.Chart().mark_text(text="Data Prep Error").properties(title="Error")


            # --- Encoding ---
            encoding = self._build_encodings(config, df_for_plot, metric_name, aggregate)
            if not encoding:
                 print("Error: Encoding generation failed. Cannot generate plot.")
                 return alt.Chart().mark_text(text="Encoding Error").properties(title="Error")


            # --- Base Chart & Mark ---
            plot_fn = getattr(config, 'plot_fn', alt.Chart.mark_point)
            if not callable(plot_fn):
                 print(f"Warning: config.plot_fn is not callable. Using default mark_point.")
                 plot_fn = alt.Chart.mark_point
            plot_fn_kwargs = getattr(config, 'plot_fn_kwargs', {}) or {}

            base_chart_obj = alt.Chart(df_for_plot) # Create base chart with data
            # Apply mark and encode
            marked_chart = plot_fn(base_chart_obj, **plot_fn_kwargs).encode(**encoding)


            # --- Structuring (Layer, Facet, Concat) ---
            # Pass the marked+encoded chart to the structuring method
            structured_chart = self._build_chart_structure(
                marked_chart, config, df_for_plot, encoding
            )


            # --- Titles and Properties ---
            chart_title = getattr(config, 'fig_title', "Chart")
            chart_properties = getattr(config, 'chart_properties', {}) or {} # Allow generic properties
            final_chart = structured_chart.properties(
                title=chart_title,
                **chart_properties # Add any other properties like width, height
            )

            # --- Interactivity ---
            if getattr(config, 'interactive_chart', False):
                final_chart = final_chart.interactive()

            # --- Display ---
            if show_chart:
                try:
                    final_chart.show()
                except Exception as show_e:
                     print(f"Error displaying chart: {show_e}")


            return final_chart

        except Exception as e:
            print(f"Error during visualization pipeline: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging
            # Return a placeholder chart indicating error
            return alt.Chart(pd.DataFrame({'error': [str(e)]})).mark_text(text='error').properties(title="Visualization Error")


# Example Usage
if __name__ == "__main__":
    # Import necessary calculation functions
    from src.analysis.bce_calculations import calculate_bce_sum, calculate_variance_on_group, calculate_pairwise_mse_on_group

    # 1. Load data and initialize the Analyzer
    try:
        # Use the provided CSV file path
        logprob_data_path = "data/test_logprobs.csv"
        analyzer = LogprobAnalyzer(logprob_data_path)
        print(f"Successfully loaded data from {logprob_data_path}")
        print("Original DataFrame head:")
        print(analyzer.original_df.head())
        print("\nOriginal columns:", analyzer.original_df.columns.tolist())

    except FileNotFoundError:
        print(f"Error: File not found at {logprob_data_path}")
        raise

    # Rename mapping for nicer labels
    analyzer.add_rename_mapping('model_name', {
        'openai-community/gpt2-medium': 'GPT-2-M',
        'meta-llama/Llama-3.2-1B': 'Llama3.2-1B'
    })
    print("\nAdded model_name rename mapping.")

    # Set sort order using the *new* names
    analyzer.set_sort_order('model_name', ['GPT-2-M', 'Llama3.2-1B'])
    print("\nSet sort order for model_name.")

    # Pre-calculate BCE sum using the updated add_categorizer
    print("\nPre-calculating BCE sum...")
    try:
        analyzer.add_categorizer(
            output_columns='_bce_sum', # Explicit output name
            source_columns=('prior_logprob', 'likelihood_logprob', 'posterior_logprob'),
            categorizer=lambda p, l, post: p + l - post if pd.notna(p) and pd.notna(l) and pd.notna(post) else pd.NA
        )
        print("BCE sum calculated and added as '_bce_sum'.")
    except Exception as e:
         print(f"Error calculating BCE sum via categorizer: {e}")
         # Manually add if categorizer fails (e.g., if columns missing in dummy data)
         if '_bce_sum' not in analyzer.current_df.columns:
              print("Attempting manual BCE sum calculation.")
              try:
                   analyzer.current_df['_bce_sum'] = calculate_bce_sum(analyzer.current_df)
                   print("Manual BCE sum calculation successful.")
              except Exception as manual_e:
                   print(f"Manual BCE sum calculation failed: {manual_e}")
                   analyzer.current_df['_bce_sum'] = 0.0 # Add dummy column


    print("\nFinal Processed DataFrame head (after adding steps & BCE sum):")
    print(analyzer.current_df.head())
    print("\nProcessed DataFrame Info:")
    analyzer.current_df.info()


    # 3. Define configurations and visualize different metrics

    # --- Plot 1: Variance BCE ---
    print("\n--- Generating Plot 1: Variance BCE ---")
    config_var = AnalysisConfig(
        plot_fn=alt.Chart.mark_bar,
        fig_title="BCE Variance by Model and Temperature",
        x_category='temperature:Q',
        y_category='bce_variance:Q', # MUST match metric_name below
        facet_category='model_name:N',
        # facet_columns=2, # Let the code calculate default columns
        tooltip_fields=[ # Explicit tooltips often better than default
            alt.Tooltip('model_name:N', title='Model'),
            alt.Tooltip('temperature:Q', title='Temp'),
            alt.Tooltip('bce_variance:Q', title='BCE Variance', format=".3f"),
            alt.Tooltip("count():Q", title="Count", format="d"),
        ],
        titles={
            'bce_variance': 'BCE (Variance method)',
            'model_name': 'Language Model',
            'temperature': 'Temperature'
        },
        interactive_chart=False,
        legend_config={"orient": "top"}
    )

    try:
        chart_var = analyzer.visualize(
            config=config_var,
            metric=calculate_variance_on_group,
            metric_name="bce_variance",
            aggregate=True,
            metric_kwargs={'value_col': '_bce_sum'} # Use metric_kwargs
        )
        chart_var.show() # Display the chart
        print("Variance BCE chart generated and displayed.")
    except Exception as e:
        print(f"Error generating Variance BCE chart: {e}")
        if '_bce_sum' not in analyzer.current_df.columns:
             print("Error hint: '_bce_sum' column not found. Check pre-calculation step.")


    # --- Plot 2: Pairwise MSE BCE ---
    print("\n--- Generating Plot 2: Pairwise MSE BCE ---")
    config_mse = AnalysisConfig(
        plot_fn=alt.Chart.mark_boxplot,
        fig_title="Pairwise BCE MSE by Model and Temperature",
        x_category='temperature:Q',
        y_category='pairwise_bce_mse:Q',
        facet_category='model_name:N',
        # facet_columns=2, # Let the code calculate default columns
        tooltip_fields=[ # Using explicit tooltips
            alt.Tooltip('model_name:N', title='Model'),
            alt.Tooltip('temperature:Q', title='Temp'),
            alt.Tooltip('pairwise_bce_mse:Q', title='Pairwise BCE MSE', format=".3f"),
            # alt.Tooltip("count():Q", title="Count", format="d"), # Count not directly applicable to boxplot points
        ],
        titles={
            'pairwise_bce_mse': 'Pairwise BCE MSE',
            'model_name': 'Language Model',
            'temperature': 'Temperature'
        },
        interactive_chart=False,
        legend_config={"orient": "top"}
    )

    try:
        chart_mse = analyzer.visualize(
            config=config_mse,
            metric=calculate_pairwise_mse_on_group,
            metric_name="pairwise_bce_mse",
            aggregate=True,
            metric_kwargs={'value_col': '_bce_sum'} # Use metric_kwargs
        )
        chart_mse.show() # Display the chart
        print("Pairwise MSE BCE chart generated and displayed.")
    except Exception as e:
        print(f"Error generating Pairwise MSE BCE chart: {e}")
        if '_bce_sum' not in analyzer.current_df.columns:
             print("Error hint: '_bce_sum' column not found. Check pre-calculation step.")

    # --- Plot 3: Raw BCE Sum (Row-wise, using default tooltips) ---
    print("\n--- Generating Plot 3: Raw BCE Sum (Row-wise, default tooltips) ---")
    config_sum_default_tt = AnalysisConfig(
        plot_fn=alt.Chart.mark_circle,
        plot_fn_kwargs={'opacity': 0.5},
        fig_title="Raw BCE Sum by Model and Temperature (Default Tooltips)",
        x_category='temperature:Q',
        y_category='bce_sum_value:Q', # MUST match metric_name below
        color_category='model_name:N', # Color by model
        tooltip_fields=None, # Explicitly request default tooltips
        titles={
            'bce_sum_value': 'BCE Sum (Row-wise)',
            'model_name': 'Language Model',
            'temperature': 'Temperature'
        },
        interactive_chart=True # Enable interactivity
    )

    try:
        # Define a simple row-wise metric function for this example
        def row_bce_sum(row):
             # Use the pre-calculated column if available, otherwise calculate on the fly
             if '_bce_sum' in row and pd.notna(row['_bce_sum']):
                 return row['_bce_sum']
             elif all(c in row for c in ['prior_logprob', 'likelihood_logprob', 'posterior_logprob']):
                 p, l, post = row['prior_logprob'], row['likelihood_logprob'], row['posterior_logprob']
                 return p + l - post if pd.notna(p) and pd.notna(l) and pd.notna(post) else pd.NA
             else:
                 return pd.NA

        chart_sum_default = analyzer.visualize(
            config=config_sum_default_tt,
            metric=row_bce_sum, # Use the wrapper
            metric_name="bce_sum_value", # MUST match config.y_category
            aggregate=False,
            # No metric_kwargs needed for this simple row function
        )
        chart_sum_default.show() # Display the chart
        print("Raw BCE Sum chart with default tooltips generated and displayed.")
    except Exception as e:
        print(f"Error generating Raw BCE Sum chart with default tooltips: {e}")