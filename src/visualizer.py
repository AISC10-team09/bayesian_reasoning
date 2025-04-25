import pandas as pd
import altair as alt
from typing import List, Optional, Callable, Dict, Any
from dataclasses import dataclass, field

@dataclass
class VisualisationConfig:
    """Configuration for visualization."""

    plot_fn: Callable[..., alt.Chart] = alt.Chart.mark_point  # Default to point plot
    fig_title: Optional[str] = None
    plot_fn_kwargs: Dict[str, Any] = field(default_factory=dict)
    chart_properties: Dict[str, Any] = field(default_factory=dict)
    x_category: Optional[str] = None  # Column for X-axis
    y_category: Optional[str] = None  # Column for Y-axis
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
    tooltip_fields: Optional[List[alt.Tooltip]] = None
    titles: Optional[Dict[str, str]] = field(default_factory=dict)  # Titles for axes/legends
    sort: Optional[Dict[str, Any]] = field(default_factory=dict)  # Sorting configuration
    scale: Optional[Dict[str, Dict[str, str]]] = field(default_factory=dict)  # Scale type configuration
    legend_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "orient": "bottom",
            "columns": 3,
            "titleAlign": "center",
            "labelLimit": 1000,
        }
    )

def visualize(data: pd.DataFrame, config: VisualisationConfig) -> alt.Chart:
    """Create a visualization based on the provided configuration.
    
    Args:
        data: DataFrame to visualize
        config: Configuration for visualization
    
    Returns:
        alt.Chart: The generated chart
    """
    if data is None or data.empty:
        print("Error: No data available for visualization.")
        return alt.Chart().mark_text(text="No Data Available").properties(title="Error")
    
    # --- Define encoding ---
    titles = config.titles or {}
    
    def get_title(category: Optional[str]) -> str:
        if not category:
            return ""
        return titles.get(category, category.replace("_", " ").title())
    
    # Basic encoding
    encoding = {}
    
    # Apply sorting configuration
    sort_config = config.sort or {}
    # Apply scale type configuration
    scale_config = config.scale or {}
    
    # Check for sorting attempts on unsupported categories
    for category_name in [config.layer_category, config.h_concat_category, config.v_concat_category]:
        if category_name and category_name in sort_config:
            print(f"Warning: Sorting for '{category_name}' via config.sort is not supported. " 
                  f"The order will follow the data's natural ordering.")
    
    if config.x_category:
        x_sort = sort_config.get(config.x_category)
        x_scale = scale_config.get(config.x_category)
        x_scale = alt.Scale(**x_scale) if x_scale else alt.Scale()
        encoding["x"] = alt.X(config.x_category, title=get_title(config.x_category), 
                             sort=x_sort, scale=x_scale)
    
    if config.y_category:
        y_sort = sort_config.get(config.y_category)
        y_scale = scale_config.get(config.y_category)
        y_scale = alt.Scale(**y_scale) if y_scale else alt.Scale()
        encoding["y"] = alt.Y(config.y_category, title=get_title(config.y_category), 
                             sort=y_sort, scale=y_scale)
    
    # Add color encoding if specified
    if config.color_category:
        color_scale = alt.Scale()
        color_cat_base = config.color_category
        color_scale = scale_config.get(color_cat_base)
        color_scale = alt.Scale(**color_scale) if color_scale else alt.Scale()
        
        if config.color_scheme:
            color_scale = alt.Scale(scheme=config.color_scheme, **color_scale) if color_scale else alt.Scale(scheme=config.color_scheme)
        if config.color_domain and config.color_range:
            color_scale = alt.Scale(domain=config.color_domain, range=config.color_range, **color_scale) if color_scale else alt.Scale(domain=config.color_domain, range=config.color_range)
        elif config.color_domain:
            color_scale = alt.Scale(domain=config.color_domain, **color_scale) if color_scale else alt.Scale(domain=config.color_domain)
        elif config.color_range:
            color_scale = alt.Scale(range=config.color_range, **color_scale) if color_scale else alt.Scale(range=config.color_range)
        
        color_sort = sort_config.get(color_cat_base)
        encoding["color"] = alt.Color(
            config.color_category,
            scale=color_scale,
            title=get_title(config.color_category),
            legend=config.color_legend or alt.Legend(**config.legend_config),
            sort=color_sort,
        )
    
    # Add additional encodings with scale type support
    if config.opacity_category:
        opacity_cat_base = config.opacity_category
        opacity_sort = sort_config.get(opacity_cat_base)
        opacity_scale = scale_config.get(opacity_cat_base)
        opacity_scale = alt.Scale(**opacity_scale) if opacity_scale else alt.Scale()
        encoding["opacity"] = alt.Opacity(
            config.opacity_category,
            title=get_title(config.opacity_category),
            legend=config.opacity_legend,
            sort=opacity_sort,
            scale=opacity_scale,
        )
    
    if config.size_category:
        size_cat_base = config.size_category
        size_sort = sort_config.get(size_cat_base)
        size_scale = scale_config.get(size_cat_base)
        size_scale = alt.Scale(**size_scale) if size_scale else alt.Scale()
        encoding["size"] = alt.Size(
            config.size_category,
            title=get_title(config.size_category),
            legend=config.size_legend,
            sort=size_sort,
            scale=size_scale,
        )
    
    if config.shape_category:
        shape_sort = sort_config.get(config.shape_category) 
        encoding["shape"] = alt.Shape(
            config.shape_category,
            title=get_title(config.shape_category),
            legend=config.shape_legend,
            sort=shape_sort,
        )
    
    if config.x_offset_category:
        x_offset_sort = sort_config.get(config.x_offset_category)
        encoding["xOffset"] = alt.XOffset(config.x_offset_category, sort=x_offset_sort)
    
    if config.y_offset_category:
        y_offset_sort = sort_config.get(config.y_offset_category)
        encoding["yOffset"] = alt.YOffset(config.y_offset_category, sort=y_offset_sort)
    
    # Add tooltips
    if config.tooltip_fields:
        encoding["tooltip"] = config.tooltip_fields
    
    # --- Create base chart ---
    plot_fn = config.plot_fn or alt.Chart.mark_point
    plot_fn_kwargs = config.plot_fn_kwargs or {}
    chart_properties = config.chart_properties or {}
    
    # Create chart with data, mark, encoding
    chart = plot_fn(alt.Chart(data), **plot_fn_kwargs).encode(**encoding)
    
    # --- Handle layering ---
    if config.layer_category:
        base_layer_cat = config.layer_category.split(":")[0]
        if base_layer_cat in data.columns:
            chart = alt.layer(
                *[
                    chart.transform_filter(f"datum.{base_layer_cat} == '{val}'")
                    for val in data[base_layer_cat].dropna().unique()
                ]
            )
    
    # --- Handle faceting ---
    if config.facet_category:
        facet_field = config.facet_category.split(":")[0]
        if facet_field in data.columns:
            facet_sort = sort_config.get(config.facet_category.split(":")[0])
            chart = chart.facet(
                facet=alt.Facet(
                    config.facet_category,
                    title=get_title(config.facet_category),
                    sort=facet_sort,
                ),
                columns=config.facet_columns or 3,
            )
    
    # --- Handle concatenation ---
    if config.h_concat_category:
        h_cat = config.h_concat_category.split(":")[0]
        if h_cat in data.columns:
            chart = alt.hconcat(
                *[
                    chart.transform_filter(
                        f"datum.{h_cat} == '{val}'"
                    ).properties(
                        title=f"{get_title(config.h_concat_category)}: {val}"
                    )
                    for val in data[h_cat].dropna().unique()
                ],
            )
    
    if config.v_concat_category:
        v_cat = config.v_concat_category.split(":")[0]
        if v_cat in data.columns:
            chart = alt.vconcat(
                *[
                    chart.transform_filter(
                        f"datum.{v_cat} == '{val}'"
                    ).properties(
                        title=f"{get_title(config.v_concat_category)}: {val}"
                    )
                    for val in data[v_cat].dropna().unique()
                ],
            )
    
    # --- Apply title and properties ---
    chart = chart.properties(title=config.fig_title, **chart_properties)
    
    return chart