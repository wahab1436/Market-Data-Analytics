"""
Dashboard Application - Plotly Dash Interface
Visualizes precomputed analysis results only
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import yaml

from .layout import create_layout
from .callbacks import register_callbacks


def create_app(config: Dict[str, Any]) -> dash.Dash:
    """Create and configure the Dash application."""
    
    # Read disclaimer from config
    disclaimer_text = """
    Market Insight Platform - Local Analytics MVP
    This is a portfolio demonstration project only.
    Not a trading system. Not for production use.
    Data may be delayed. Past performance is not indicative of future results.
    """
    
    # Initialize Dash app
    app = dash.Dash(
        __name__,
        title="Market Insight Platform",
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
        ],
        suppress_callback_exceptions=True
    )
    
    # Apply custom CSS
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f8f9fa;
                    color: #212529;
                }
                .dashboard-container {
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .navbar {
                    background: linear-gradient(135deg, #2E86AB 0%, #1a5276 100%);
                    color: white;
                    padding: 1rem 2rem;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                .footer {
                    background-color: #343a40;
                    color: #adb5bd;
                    padding: 2rem;
                    margin-top: 3rem;
                    font-size: 0.9rem;
                    border-top: 1px solid #495057;
                }
                .kpi-card {
                    background: white;
                    border-radius: 8px;
                    padding: 1.5rem;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    height: 100%;
                    border-left: 4px solid #2E86AB;
                }
                .insight-card {
                    background: white;
                    border-radius: 8px;
                    padding: 1.5rem;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    margin-bottom: 1rem;
                    border-left: 4px solid #18A558;
                }
                .chart-container {
                    background: white;
                    border-radius: 8px;
                    padding: 1.5rem;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    margin-bottom: 1.5rem;
                }
                .loading-container {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 400px;
                }
                .last-update {
                    font-size: 0.85rem;
                    color: #6c757d;
                    margin-top: 0.5rem;
                }
                .positive {
                    color: #18A558;
                    font-weight: 600;
                }
                .negative {
                    color: #C73E1D;
                    font-weight: 600;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Set app layout
    app.layout = create_layout(config, disclaimer_text)
    
    # Register callbacks
    register_callbacks(app, config)
    
    return app


if __name__ == "__main__":
    # Load configuration
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create and run app
    app = create_app(config)
    app.run_server(
        debug=False,
        host='127.0.0.1',
        port=8050,
        dev_tools_ui=False,
        dev_tools_props_check=False
    )
