import pandas as pd
import numpy as np
from typing import Callable

class Analyzer:
    """Simple data processing and analysis for log probability data.
    
    Each method applies transformations immediately to the dataframe.
    """
    
    def __init__(self, source: pd.DataFrame | str):
        """Initialize with either a DataFrame or path to CSV file."""
        if isinstance(source, str):
            try:
                self.df = pd.read_csv(source)
            except FileNotFoundError:
                raise FileNotFoundError(f"CSV file not found: {source}")
            except Exception as e:
                raise ValueError(f"Error reading CSV: {e}")
        elif isinstance(source, pd.DataFrame):
            self.df = source.copy()
        else:
            raise ValueError("Source must be a DataFrame or file path string")
    
    def filter(self, filter_spec: dict[str, list | Callable] | Callable):
        """Filter dataframe rows based on conditions.
        
        Args:
            filter_spec: Either:
                - dict mapping columns to allowed values: {"column": [val1, val2]}
                - dict mapping columns to filter functions: {"column": lambda x: x > 0}
                - callable that takes DataFrame and returns filtered DataFrame
        
        Returns:
            self: For method chaining
        """
        if callable(filter_spec):
            # Apply callable directly to dataframe
            self.df = filter_spec(self.df)
        elif isinstance(filter_spec, dict):
            for col, condition in filter_spec.items():
                if col not in self.df.columns:
                    print(f"Warning: Column '{col}' not found. Skipping filter.")
                    continue
                
                if callable(condition):
                    # Apply function to column
                    self.df = self.df[condition(self.df[col])]
                elif isinstance(condition, list):
                    # Filter to values in list
                    self.df = self.df[self.df[col].isin(condition)]
                else:
                    print(f"Warning: Invalid filter condition type for '{col}'. Skipping.")
        else:
            raise ValueError("filter_spec must be a dictionary or callable function")
            
        return self
    
    def rename(self, rename_spec: dict[str, str | dict[str, str] | Callable] | Callable):
        """Rename columns or remap column values.
        
        Args:
            rename_spec: Either:
                - dict mapping old column names to new names: {"old_col": "new_col"}
                - dict mapping columns to value mapping dicts: {"col": {"old_val": "new_val"}}
                - dict mapping columns to transformation functions: {"col": lambda x: x.upper()}
                - callable that takes DataFrame and returns renamed DataFrame
        
        Returns:
            self: For method chaining
        """
        if callable(rename_spec):
            # Apply callable directly to dataframe
            self.df = rename_spec(self.df)
        elif isinstance(rename_spec, dict):
            for col, spec in rename_spec.items():
                if col not in self.df.columns:
                    print(f"Warning: Column '{col}' not found. Skipping rename.")
                    continue
                
                if isinstance(spec, str):
                    # Rename column
                    self.df = self.df.rename(columns={col: spec})
                elif isinstance(spec, dict):
                    # Remap values using dictionary
                    self.df[col] = self.df[col].map(spec).fillna(self.df[col])
                elif callable(spec):
                    # Apply function to column
                    self.df[col] = self.df[col].apply(spec)
                else:
                    print(f"Warning: Invalid rename spec type for '{col}'. Skipping.")
        else:
            raise ValueError("rename_spec must be a dictionary or callable function")
            
        return self
    
    def add_column(self, column_name: str, column_spec: dict[str, dict[str, str] | Callable] | Callable):
        """Add a new column to the dataframe.
        
        Args:
            column_name: Name of the new column
            column_spec: Either:
                - callable that takes DataFrame and returns a Series
                - dict mapping source columns to transformations:
                  {"source_col": lambda x: x.upper()} or
                  {"source_col": {"old_val": "new_val"}}
        
        Returns:
            self: For method chaining
        """
        if callable(column_spec):
            # Apply function to dataframe to generate column
            self.df[column_name] = column_spec(self.df)
        elif isinstance(column_spec, dict):
            # Process each source column specification
            for source_col, spec in column_spec.items():
                if source_col not in self.df.columns:
                    print(f"Warning: Source column '{source_col}' not found. Skipping.")
                    continue
                
                if callable(spec):
                    # Apply function to transform the source column
                    self.df[column_name] = self.df[source_col].apply(spec)
                elif isinstance(spec, dict):
                    # Use mapping to transform values
                    self.df[column_name] = self.df[source_col].map(spec)
                else:
                    print(f"Warning: Invalid specification type for '{source_col}'. Skipping.")
        else:
            raise ValueError("column_spec must be a dictionary or callable function")
        
        return self
    
    def sort(self, sort_spec: list[str] | dict[str, list | Callable] | Callable):
        """Sort the dataframe.
        
        Args:
            sort_spec: Either:
                - list of column names to sort by
                - dict mapping columns to sort order: {"column": [val1, val2]}
                - dict mapping columns to sort functions: {"column": lambda x: x.lower()}
                - callable that takes DataFrame and returns sorted DataFrame
        
        Returns:
            self: For method chaining
        """
        if callable(sort_spec):
            # Apply callable directly to dataframe
            self.df = sort_spec(self.df)
        elif isinstance(sort_spec, list):
            # Sort by columns in list
            missing_cols = [col for col in sort_spec if col not in self.df.columns]
            if missing_cols:
                print(f"Warning: Sort columns not found: {missing_cols}")
                sort_cols = [col for col in sort_spec if col in self.df.columns]
                if not sort_cols:
                    return self
                self.df = self.df.sort_values(sort_cols)
            else:
                self.df = self.df.sort_values(sort_spec)
        elif isinstance(sort_spec, dict):
            temp_cols = []
            
            for col, spec in sort_spec.items():
                if col not in self.df.columns:
                    print(f"Warning: Column '{col}' not found. Skipping sort.")
                    continue
                
                if callable(spec):
                    # Create temporary column with transformed values for sorting
                    temp_col = f"__{col}_sort_temp__"
                    temp_cols.append(temp_col)
                    self.df[temp_col] = self.df[col].apply(spec)
                elif isinstance(spec, list):
                    # Create a mapping dictionary from value to position in the list
                    # This ensures proper sorting based on the specified order
                    order_map = {val: i for i, val in enumerate(spec)}
                    
                    # Create temporary column with position values for sorting
                    temp_col = f"__{col}_sort_temp__"
                    temp_cols.append(temp_col)
                    self.df[temp_col] = self.df[col].map(order_map)
                    
                    # Handle values not in the specification
                    self.df[temp_col] = self.df[temp_col].fillna(len(spec))
                else:
                    print(f"Warning: Invalid sort spec type for '{col}'. Skipping.")
            
            # Sort by all temporary columns
            sort_cols = [f"__{col}_sort_temp__" for col, spec in sort_spec.items() 
                        if col in self.df.columns and (callable(spec) or isinstance(spec, list))]
            
            if sort_cols:
                self.df = self.df.sort_values(sort_cols, na_position="last")
            
            # Remove temporary columns
            if temp_cols:
                self.df = self.df.drop(columns=temp_cols)

        else:
            raise ValueError("sort_spec must be a list, dictionary, or callable")
            
        return self
    
    def calculate_metric(
        self,
        metric_name: str,
        metric_func: Callable,
        metric_col: str | None = None,
        group_by_cols: list[str] | None = None, 
        inherit_identical_values: bool = False,
        **metric_kwargs
    ) -> 'Analyzer':
        """Calculate metrics on the dataframe.
        
        Args:
            metric_name: Name for the calculated metric column
            metric_func: Function to calculate the metric. If group_by_cols and metric_col
                         are provided, this function operates on the specified column Series
                         within each group. Otherwise, it operates on the entire group DataFrame
                         (if group_by_cols is provided) or the entire row (if group_by_cols is None).
            metric_col: Optional column name to apply the metric_func to directly within groups.
                        Ignored if group_by_cols is None.
            group_by_cols: Columns to group by (None for row-wise calculation)
            inherit_identical_values: If True, columns with identical values within groups are preserved
            **metric_kwargs: Additional arguments for metric function
        
        Returns:
            Analyzer instance with calculated metrics
        """
        if self.df.empty:
            print("Warning: DataFrame is empty. Cannot calculate metrics.")
            return Analyzer(pd.DataFrame())
        
        result_df = self.df.copy()
        
        # Row-wise calculation (no grouping) - metric_col is ignored here
        if group_by_cols is None:
            try:
                result_df[metric_name] = result_df.apply(
                    lambda row: metric_func(row, **metric_kwargs), axis=1
                )
                return Analyzer(result_df)
            except Exception as e:
                print(f"Error in row-wise calculation: {e}")
                return Analyzer(pd.DataFrame())
        
        # Grouped calculation
        try:
            # Validate grouping columns
            missing_cols = [col for col in group_by_cols if col not in result_df.columns]
            if missing_cols:
                print(f"Error: Missing grouping columns: {missing_cols}")
                return Analyzer(pd.DataFrame())
            
            # Perform grouping and apply metric function
            grouped = result_df.groupby(group_by_cols, observed=True)
            
            # Apply metric function based on whether metric_col is specified
            if metric_col:
                if metric_col not in result_df.columns:
                    print(f"Error: Metric column '{metric_col}' not found.")
                    return Analyzer(pd.DataFrame())
                # Apply function to the specified column within each group
                result = grouped[metric_col].agg(metric_func, **metric_kwargs)
            else:
                # Apply function to the entire group DataFrame
                result = grouped.apply(lambda g: metric_func(g, **metric_kwargs))
            
            # Handle list-returning metrics
            if isinstance(result.iloc[0] if not result.empty else None, (list, np.ndarray)):
                result = result.explode()
                result = pd.to_numeric(result, errors="coerce")
            
            # Create result dataframe with the calculated metric
            result.name = metric_name
            result_df = result.reset_index()
            
            # Inherit columns with identical values within each group
            if inherit_identical_values and not result.empty:
                # For each non-groupby column, check if all values in each group are identical
                cols_to_check = [c for c in self.df.columns if c not in group_by_cols]
                
                # Use nunique to find columns that have only one unique value per group
                nunique_df = grouped[cols_to_check].nunique()
                
                # Get columns that have only 1 unique value for all groups
                identical_cols = [col for col in cols_to_check if (nunique_df[col] <= 1).all()]
                
                if identical_cols:
                    # Get the first value for each identical column in each group
                    identical_values = grouped[identical_cols].first().reset_index()
                    
                    # Merge with result_df to add these columns
                    result_df = pd.merge(result_df, identical_values, on=group_by_cols)
            
            return Analyzer(result_df)
            
        except Exception as e:
            print(f"Error in grouped calculation: {e}")
            import traceback
            traceback.print_exc()
            return Analyzer(pd.DataFrame())