"""
Dashboard page for exploring FLFP dataset
Contains sidebar filters and three visualizations
"""

from dash import html, dcc, Input, Output, State, callback
import dash_mantine_components as dmc
from utils.data_loader import load_flfp_data, get_data_summary, get_indicator_options
from components.column_chart import create_column_chart
from components.sunburst import create_sunburst
from components.line_chart import create_line_chart
from utils.modeling import get_display_year

def create_dashboard_layout():
    """
    Create the dashboard page layout with sidebar and visualizations.
    
    Returns:
        dmc.Container: The complete dashboard layout
    """
    # Load data summary for filter options
    summary = get_data_summary()
    
    # Get unique values for filters
    years = summary.get("years", [])
    countries = summary.get("countries", [])
    regions = summary.get("regions", [])
    indicators = get_indicator_options()
    
    # Initial data load
    df = load_flfp_data()
    
    # Create initial visualizations with default indicator
    initial_indicator = indicators[0]["value"] if indicators else None
    initial_viz_1 = create_column_chart(df, initial_indicator, "all_regions", None, None)
    initial_viz_2 = create_sunburst(df, initial_indicator, "all_regions", None, None)
    initial_viz_3 = create_line_chart(df, initial_indicator, "all_regions", None, None)
    
    # Model years for the About text (training ends the year before the display year)
    display_year = get_display_year(df)
    first_year = int(df['year'].min())
    
    return dmc.Container(
        [
            dmc.Grid(
                [
                    # Sidebar column
                    dmc.GridCol(
                        [
                            dmc.Paper(
                                [
                                    # App title
                                    dmc.Title(
                                        "Female Labor Force Participation Explorer",
                                        order=2,
                                        mb="sm",
                                        style={"lineHeight": 1.2}
                                    ),
                                    dmc.Divider(mb="md"),
                                    
                                    dmc.Title("Filters", order=3, mb="md"),
                                    
                                    # Indicator selector
                                    dmc.Text("Indicator", size="sm", fw=500, mb="xs"),
                                    dmc.Select(
                                        id="indicator-selector",
                                        data=indicators,
                                        value=initial_indicator,
                                        placeholder="Select an indicator",
                                        clearable=False,
                                        searchable=True,
                                        mb="lg"
                                    ),
                                    
                                    # Region radio selector
                                    dmc.Text("Region", size="sm", fw=500, mb="xs"),
                                    dmc.RadioGroup(
                                        id="region-selector",
                                        children=[dmc.Radio(label="All Regions", value="All Regions")] + 
                                                 [dmc.Radio(label=r, value=r) for r in regions],
                                        value="All Regions",
                                        mb="lg"
                                    ),
                                    
                                    # Country multi-select
                                    dmc.Text("Countries", size="sm", fw=500, mb="xs"),
                                    dmc.MultiSelect(
                                        id="country-selector",
                                        data=[{"value": c, "label": c} for c in countries],
                                        placeholder="Select countries",
                                        clearable=True,
                                        searchable=True,
                                        mb="lg"
                                    ),
                                    
                                    # Year range selector
                                    dmc.Text("Year Range", size="sm", fw=500, mb="xs"),
                                    dmc.RangeSlider(
                                        id="year-range-slider",
                                        min=min(years) if years else 2000,
                                        max=max(years) if years else 2023,
                                        value=[min(years) if years else 2000, max(years) if years else 2023],
                                        marks=[
                                            {"value": year, "label": str(year)}
                                            for year in [min(years) if years else 2000, max(years) if years else 2023]]
                                        if years else [],
                                        mb="xl"
                                    ),
                                    
                                    # Description
                                    dmc.Divider(mb="md"),
                                    dmc.Text(
                                        "About this App",
                                        size="sm",
                                        fw=600,
                                        mb="xs"
                                    ),
                                    dmc.Text(
                                        [
                                            "Use the Dashboard to explore patterns and trends in female labor force participation (FLFP) ",
                                            "and related socioeconomic indicators across countries and regions."
                                        ],
                                        size="xs",
                                        c="dimmed",
                                        mb="xs",
                                        style={"lineHeight": 1.5}
                                    ),
                                    dmc.Text(
                                        [
                                            "Then, go to the Model Explorer to see what a predictive model says about how these factors ",
                                            "relate to FLFP. Click a country on the map to load its values, or move the sliders to create ",
                                            "your own hypothetical scenarios, and see where the predicted rate falls among actual rates ",
                                            "around the world."
                                        ],
                                        size="xs",
                                        c="dimmed",
                                        mb="xs",
                                        style={"lineHeight": 1.5}
                                    ),
                                    dmc.Text(
                                        [
                                            "The predictive model is a random forest of 200 decision trees, trained on country-year data ",
                                            f"from {first_year} to {display_year - 1} using the indicators featured in the explorer. ",
                                            f"The map and histogram show each country's actual FLFP in {display_year}."
                                        ],
                                        size="xs",
                                        c="dimmed",
                                        mb="xs",
                                        style={"lineHeight": 1.5}
                                    ),
                                    dmc.Text(
                                        [
                                            "All data come from the World Bank's World Development Indicators and Worldwide Governance ",
                                            "Indicators. The code is available on ",
                                            dmc.Anchor(
                                                "GitHub",
                                                href="https://github.com/eteitelbaum/flfp-explorer",
                                                target="_blank",
                                                size="xs"
                                            ),
                                            "."
                                        ],
                                        size="xs",
                                        c="dimmed",
                                        style={"lineHeight": 1.5}
                                    ),
                                    
                        ],
                        p="md",
                        withBorder=True,
                        style={
                            "height": "87.5vh",
                            "overflowY": "auto",
                            "overflowX": "hidden"
                        }
                    )
                ],
                span=3
            ),                    # Main content column
                    dmc.GridCol(
                        [
                            # First row: Two equally sized visualizations
                            dmc.Grid(
                                [
                                    dmc.GridCol(
                                        [
                                            dmc.Paper(
                                                [
                                                    html.Div(
                                                        initial_viz_1,
                                                        id="viz-1-container",
                                                        style={
                                                            "height": "100%",
                                                            "overflowY": "auto",
                                                            "overflowX": "hidden"
                                                        }
                                                    )
                                                ],
                                                p="md",
                                                withBorder=True,
                                                style={"height": "40vh"}
                                            )
                                        ],
                                        span=6
                                    ),
                                    dmc.GridCol(
                                        [
                                            dmc.Paper(
                                                [
                                                    html.Div(
                                                        initial_viz_2,
                                                        id="viz-2-container",
                                                        style={"height": "100%"}
                                                    )
                                                ],
                                                p="md",
                                                withBorder=True,
                                                style={"height": "40vh"}
                                            )
                                        ],
                                        span=6
                                    ),
                                ],
                                gutter="md",
                                mb="md"
                            ),
                            
                            # Second row: One full-width visualization
                            dmc.Grid(
                                [
                                    dmc.GridCol(
                                        [
                                            dmc.Paper(
                                                [
                                                    html.Div(
                                                        initial_viz_3,
                                                        id="viz-3-container",
                                                        style={"height": "100%"}
                                                    )
                                                ],
                                                p="md",
                                                withBorder=True,
                                                style={"height": "45vh"}
                                            )
                                        ],
                                        span=12
                                    ),
                                ],
                                gutter="md"
                            ),
                        ],
                        span=9
                    ),
                ],
                gutter="md"
            ),
            
            # Store filtered data and sunburst selection state
            dcc.Store(id="filtered-data-store", data=df.to_dict("records")),
            dcc.Store(id="sunburst-selection-store", data=None),
        ],
        fluid=True,
        style={"padding": "20px"}
    )

@callback(
    [Output("viz-1-container", "children"),
     Output("viz-2-container", "children"),
     Output("viz-3-container", "children"),
     Output("sunburst-selection-store", "data")],
    [Input("year-range-slider", "value"),
     Input("region-selector", "value"),
     Input("country-selector", "value"),
     Input("indicator-selector", "value"),
     Input("viz-2", "clickData")],
    [State("sunburst-selection-store", "data")]
)
def update_dashboard(year_range, selected_region, selected_countries, selected_indicator, 
                     sunburst_click, sunburst_store):
    """
    Update the dashboard based on filter selections with priority logic.
    
    Selection Priority:
        1. Sunburst click (highest) - overrides all other selections
        2. Country dropdown - overrides region radio
        3. Region radio - applies when no countries selected
        4. Default - "All Regions"
    
    Args:
        year_range: List of [min_year, max_year]
        selected_region: Single selected region from radio (or "All Regions")
        selected_countries: List of selected country names from multi-select
        selected_indicator: Selected indicator column name
        sunburst_click: Click data from sunburst chart
        sunburst_store: Previous sunburst selection state
        
    Returns:
        tuple: (viz_1, viz_2, viz_3, sunburst_store)
    """
    import pandas as pd
    from dash import ctx
    
    # Load and filter data by year range
    df = load_flfp_data()
    
    if year_range:
        df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    
    # Determine selection state based on priority logic
    mode, effective_region, effective_countries, new_sunburst_store = _determine_selection_mode(
        selected_region, selected_countries, sunburst_click, sunburst_store, df
    )
    
    # Create visualizations with determined mode and selections
    viz_1 = create_column_chart(df, selected_indicator, mode, effective_region, effective_countries)
    viz_2 = create_sunburst(df, selected_indicator, mode, effective_region, effective_countries)
    viz_3 = create_line_chart(df, selected_indicator, mode, effective_region, effective_countries)
    
    return viz_1, viz_2, viz_3, new_sunburst_store


def _determine_selection_mode(region_radio, country_dropdown, sunburst_click, sunburst_store, df):
    """
    Determine the display mode and effective selections based on priority logic.
    
    Returns:
        tuple: (mode, effective_region, effective_countries, new_sunburst_store)
    """
    from dash import ctx
    
    # Check if sunburst was clicked
    triggered_id = ctx.triggered_id if ctx.triggered else None
    
    if triggered_id == "viz-2" and sunburst_click:
        # Sunburst click has highest priority
        clicked_label = sunburst_click['points'][0]['label']
        
        # Determine if it's a region or country
        if clicked_label == "World":
            # Clicked root, reset to all regions
            return "all_regions", None, None, None
        
        # Check if clicked item is a region or country
        regions = df['region'].dropna().unique().tolist()
        
        if clicked_label in regions:
            # Clicked a region
            return "region_only", clicked_label, None, {"type": "region", "value": clicked_label}
        else:
            # Clicked a country
            return "single_country", None, [clicked_label], {"type": "country", "value": clicked_label}
    
    # If no new sunburst click, check if we should preserve previous sunburst selection
    # Only preserve if no dropdown selections are active
    if sunburst_store and not country_dropdown:
        store_type = sunburst_store.get("type")
        store_value = sunburst_store.get("value")
        
        if store_type == "region" and region_radio == "All Regions":
            return "region_only", store_value, None, sunburst_store
        elif store_type == "country" and region_radio == "All Regions":
            return "single_country", None, [store_value], sunburst_store
    
    # Clear sunburst store if dropdown selections are active
    if country_dropdown or (region_radio and region_radio != "All Regions"):
        sunburst_store = None
    
    # Check country dropdown (second priority)
    if country_dropdown and len(country_dropdown) > 0:
        if len(country_dropdown) == 1:
            return "single_country", None, country_dropdown, None
        else:
            return "multi_country", None, country_dropdown, None
    
    # Check region radio (third priority)
    if region_radio and region_radio != "All Regions":
        return "region_only", region_radio, None, None
    
    # Default: all regions
    return "all_regions", None, None, None

