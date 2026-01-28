"""
Dashboard Layout Module
Professional financial dashboard layout with Plotly Dash
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import datetime
from typing import Dict, Any, List


def create_layout(config: Dict[str, Any], disclaimer_text: str) -> html.Div:
    """Create the main dashboard layout."""
    
    symbols = config['data']['symbols']
    color_palette = config['dashboard']['color_palette']
    
    # Current timestamp for display
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Header with navigation
    header = dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Market Insight Platform", className="navbar-title"),
                    html.P("Local Analytics MVP | Portfolio Demonstration", 
                          className="navbar-subtitle")
                ], width="auto"),
            ], align="center", className="g-0"),
            
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            
            dbc.Collapse(
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Price Analysis", href="#price-section")),
                    dbc.NavItem(dbc.NavLink("Volatility", href="#volatility-section")),
                    dbc.NavItem(dbc.NavLink("Volume", href="#volume-section")),
                    dbc.NavItem(dbc.NavLink("Similarity", href="#similarity-section")),
                    dbc.NavItem(dbc.NavLink("Models", href="#models-section")),
                    dbc.NavItem(dbc.NavLink("Explainability", href="#explainability-section")),
                ], className="ms-auto", navbar=True),
                id="navbar-collapse",
                navbar=True,
            ),
        ], fluid=True),
        color="primary",
        dark=True,
        className="mb-4",
        style={"boxShadow": "0 2px 10px rgba(0,0,0,0.1)"}
    )
    
    # Symbol selector and controls
    controls_card = dbc.Card([
        dbc.CardHeader("Controls", className="fw-bold"),
        dbc.CardBody([
            html.Div([
                html.Label("Select Asset:", className="form-label"),
                dcc.Dropdown(
                    id="symbol-selector",
                    options=[{"label": s, "value": s} for s in symbols],
                    value=symbols[0] if symbols else None,
                    clearable=False,
                    className="mb-3"
                ),
                
                html.Label("Date Range:", className="form-label"),
                dcc.DatePickerRange(
                    id="date-range",
                    start_date=config['data']['date_range']['start'],
                    end_date=config['data']['date_range']['end'],
                    display_format='YYYY-MM-DD',
                    className="mb-3"
                ),
                
                html.Label("Analysis Focus:", className="form-label"),
                dcc.RadioItems(
                    id="analysis-focus",
                    options=[
                        {'label': 'Price & Trends', 'value': 'price'},
                        {'label': 'Risk & Volatility', 'value': 'volatility'},
                        {'label': 'Volume & Liquidity', 'value': 'volume'},
                        {'label': 'Patterns & Similarity', 'value': 'similarity'}
                    ],
                    value='price',
                    className="mb-3"
                ),
                
                dbc.Button(
                    "Refresh View",
                    id="refresh-button",
                    color="primary",
                    className="w-100",
                    n_clicks=0
                ),
            ])
        ]),
        dbc.CardFooter(f"Last Updated: {current_time}", className="text-muted small")
    ], className="h-100")
    
    # KPI Cards (will be populated dynamically)
    kpi_cards = dbc.Row([
        dbc.Col(create_kpi_card("Current Price", "$--", "primary", "price-kpi"), width=3),
        dbc.Col(create_kpi_card("Daily Return", "--%", "success", "return-kpi"), width=3),
        dbc.Col(create_kpi_card("20-day Vol", "--%", "warning", "volatility-kpi"), width=3),
        dbc.Col(create_kpi_card("Avg Volume", "--", "info", "volume-kpi"), width=3),
    ], className="g-3 mb-4")
    
    # Main content sections
    content = html.Div([
        # Price Analysis Section
        html.Div([
            html.H2("Price Analysis", className="section-title", id="price-section"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        id="loading-price-chart",
                        type="circle",
                        children=dcc.Graph(id="price-chart", className="chart-container")
                    )
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Price Insights", className="fw-bold"),
                        dbc.CardBody(id="price-insights")
                    ], className="h-100")
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Trend Metrics", className="fw-bold"),
                        dbc.CardBody(id="trend-metrics")
                    ], className="h-100")
                ], width=6)
            ], className="g-3 mt-3")
        ], className="section-container mb-5"),
        
        # Volatility Analysis Section
        html.Div([
            html.H2("Volatility Analysis", className="section-title", id="volatility-section"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        id="loading-volatility-chart",
                        type="circle",
                        children=dcc.Graph(id="volatility-chart", className="chart-container")
                    )
                ], width=8),
                dbc.Col([
                    dcc.Loading(
                        id="loading-volatility-distribution",
                        type="circle",
                        children=dcc.Graph(id="volatility-distribution-chart", className="chart-container")
                    )
                ], width=4)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Volatility Insights", className="fw-bold"),
                        dbc.CardBody(id="volatility-insights")
                    ], className="h-100")
                ], width=12)
            ], className="g-3 mt-3")
        ], className="section-container mb-5"),
        
        # Volume Analysis Section
        html.Div([
            html.H2("Volume Analysis", className="section-title", id="volume-section"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        id="loading-volume-chart",
                        type="circle",
                        children=dcc.Graph(id="volume-chart", className="chart-container")
                    )
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Volume-Price Relationship", className="fw-bold"),
                        dbc.CardBody(id="volume-relationship")
                    ], className="h-100")
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Volume Insights", className="fw-bold"),
                        dbc.CardBody(id="volume-insights")
                    ], className="h-100")
                ], width=6)
            ], className="g-3 mt-3")
        ], className="section-container mb-5"),
        
        # Similarity Analysis Section
        html.Div([
            html.H2("Pattern Similarity", className="section-title", id="similarity-section"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        id="loading-similarity-chart",
                        type="circle",
                        children=dcc.Graph(id="similarity-chart", className="chart-container")
                    )
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Historical Analogs", className="fw-bold"),
                        dbc.CardBody(id="analog-insights")
                    ], className="h-100")
                ], width=12)
            ], className="g-3 mt-3")
        ], className="section-container mb-5"),
        
        # Machine Learning Models Section
        html.Div([
            html.H2("Model Analysis", className="section-title", id="models-section"),
            html.Hr(),
            dbc.Tabs([
                dbc.Tab(label="Regression Models", tab_id="regression-tab"),
                dbc.Tab(label="KNN Similarity", tab_id="knn-tab"),
                dbc.Tab(label="XGBoost Predictions", tab_id="xgboost-tab"),
            ], id="model-tabs", active_tab="regression-tab", className="mb-3"),
            
            html.Div(id="model-tab-content"),
        ], className="section-container mb-5"),
        
        # Explainability Section
        html.Div([
            html.H2("Model Explainability", className="section-title", id="explainability-section"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        id="loading-explainability-chart",
                        type="circle",
                        children=dcc.Graph(id="explainability-chart", className="chart-container")
                    )
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Model Insights", className="fw-bold"),
                        dbc.CardBody(id="model-insights")
                    ], className="h-100")
                ], width=12)
            ], className="g-3 mt-3")
        ], className="section-container mb-5"),
        
        # Cross-Symbol Analysis Section
        html.Div([
            html.H2("Cross-Symbol Analysis", className="section-title", id="cross-symbol-section"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        id="loading-correlation-chart",
                        type="circle",
                        children=dcc.Graph(id="correlation-chart", className="chart-container")
                    )
                ], width=6),
                dbc.Col([
                    dcc.Loading(
                        id="loading-performance-chart",
                        type="circle",
                        children=dcc.Graph(id="performance-chart", className="chart-container")
                    )
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Cross-Symbol Insights", className="fw-bold"),
                        dbc.CardBody(id="cross-symbol-insights")
                    ], className="h-100")
                ], width=12)
            ], className="g-3 mt-3")
        ], className="section-container mb-5"),
    ])
    
    # Footer with disclaimer
    footer = dbc.Container([
        html.Hr(),
        html.Div([
            html.H5("Disclaimer", className="mb-3"),
            html.P(disclaimer_text, className="small"),
            html.P([
                "This dashboard is for demonstration purposes only. ",
                html.Strong("Not financial advice."),
                " Use at your own risk."
            ], className="small text-muted mt-2"),
            html.Div([
                html.Span("Market Insight Platform v1.0.0", className="me-3"),
                html.Span("Local-First MVP", className="me-3"),
                html.Span("Python 3.13 | Plotly Dash", className="me-3"),
                html.Span(f"Data as of: {current_time}")
            ], className="small text-muted mt-3")
        ], className="py-4")
    ], fluid=True)
    
    # Main layout
    layout = html.Div([
        # Store preloaded data
        dcc.Store(id='analysis-data-store'),
        dcc.Store(id='model-data-store'),
        dcc.Store(id='featured-data-store'),
        
        # Header
        header,
        
        # Main container
        dbc.Container([
            dbc.Row([
                # Sidebar controls
                dbc.Col(controls_card, width=3, className="sidebar"),
                
                # Main content
                dbc.Col([
                    # KPI Cards
                    kpi_cards,
                    
                    # Main content
                    content
                ], width=9)
            ]),
            
            # Footer
            footer
            
        ], fluid=True, className="dashboard-container")
    ])
    
    return layout


def create_kpi_card(title: str, value: str, color: str, kpi_id: str) -> dbc.Card:
    """Create a KPI card component."""
    color_map = {
        "primary": "#2E86AB",
        "success": "#18A558",
        "warning": "#F18F01",
        "danger": "#C73E1D",
        "info": "#A23B72"
    }
    
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="card-subtitle mb-2 text-muted"),
            html.H3(id=kpi_id, children=value, className="card-title fw-bold"),
            html.Div(id=f"{kpi_id}-trend", className="small")
        ])
    ], className="text-center border-0 shadow-sm", 
       style={"borderLeft": f"4px solid {color_map.get(color, color_map['primary'])}"})


def create_metric_card(title: str, value: Any, change: str = None) -> html.Div:
    """Create a metric display card."""
    card_content = [
        html.H6(title, className="metric-title"),
        html.H4(value, className="metric-value")
    ]
    
    if change:
        trend_class = "positive" if change.startswith("+") else "negative"
        card_content.append(html.Span(change, className=f"metric-change {trend_class}"))
    
    return html.Div(card_content, className="metric-card")


def create_insight_card(insights: List[str]) -> html.Div:
    """Create an insight display card."""
    if not insights:
        return html.Div("No insights available.", className="text-muted")
    
    insight_items = []
    for i, insight in enumerate(insights):
        insight_items.append(html.Li(insight, className="insight-item"))
        if i < len(insights) - 1:
            insight_items.append(html.Hr(className="my-2"))
    
    return html.Ul(insight_items, className="insight-list")


def create_model_tab_content(tab_id: str) -> html.Div:
    """Create content for model tabs."""
    if tab_id == "regression-tab":
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        children=dcc.Graph(id="regression-chart", className="chart-container")
                    )
                ], width=8),
                dbc.Col([
                    dcc.Loading(
                        children=dcc.Graph(id="regression-metrics-chart", className="chart-container")
                    )
                ], width=4)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Regression Insights", className="fw-bold"),
                        dbc.CardBody(id="regression-insights")
                    ])
                ], width=12)
            ], className="mt-3")
        ])
    
    elif tab_id == "knn-tab":
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        children=dcc.Graph(id="knn-chart", className="chart-container")
                    )
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("KNN Analogs", className="fw-bold"),
                        dbc.CardBody(id="knn-analogs")
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("KNN Insights", className="fw-bold"),
                        dbc.CardBody(id="knn-insights")
                    ])
                ], width=6)
            ], className="mt-3")
        ])
    
    elif tab_id == "xgboost-tab":
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        children=dcc.Graph(id="xgboost-prediction-chart", className="chart-container")
                    )
                ], width=8),
                dbc.Col([
                    dcc.Loading(
                        children=dcc.Graph(id="xgboost-importance-chart", className="chart-container")
                    )
                ], width=4)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("XGBoost Performance", className="fw-bold"),
                        dbc.CardBody(id="xgboost-metrics")
                    ])
                ], width=12)
            ], className="mt-3")
        ])
    
    return html.Div("Select a model tab.", className="text-muted")


# CSS styles inline (can also be in separate CSS file)
styles = """
.dashboard-container {
    padding: 20px;
}

.section-container {
    background: white;
    border-radius: 8px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.section-title {
    color: #2E86AB;
    margin-bottom: 16px;
    font-weight: 600;
}

.chart-container {
    background: white;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.kpi-card {
    text-align: center;
    padding: 16px;
    border-radius: 8px;
    background: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    border-left: 4px solid #2E86AB;
}

.kpi-value {
    font-size: 24px;
    font-weight: 700;
    margin: 8px 0;
}

.kpi-label {
    font-size: 14px;
    color: #6c757d;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.insight-list {
    list-style-type: none;
    padding-left: 0;
    margin-bottom: 0;
}

.insight-item {
    padding: 8px 0;
    color: #495057;
    font-size: 14px;
    line-height: 1.5;
}

.insight-item:not(:last-child) {
    border-bottom: 1px solid #e9ecef;
}

.metric-card {
    padding: 12px;
    background: #f8f9fa;
    border-radius: 6px;
    margin-bottom: 8px;
}

.metric-title {
    font-size: 12px;
    color: #6c757d;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}

.metric-value {
    font-size: 18px;
    font-weight: 600;
    color: #212529;
    margin-bottom: 4px;
}

.metric-change {
    font-size: 12px;
    font-weight: 500;
}

.metric-change.positive {
    color: #18A558;
}

.metric-change.negative {
    color: #C73E1D;
}

.sidebar {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 16px;
}

.navbar-title {
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 0;
}

.navbar-subtitle {
    font-size: 14px;
    color: rgba(255,255,255,0.8);
    margin-bottom: 0;
}
"""
