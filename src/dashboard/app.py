"""
Dashboard Application - Plotly Dash Interface
Professional market analytics dashboard with comprehensive data visualization
"""

import dash
from dash import dcc, html
from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import yaml
import logging

from .layout import create_layout
from .callbacks import register_callbacks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app(config: Dict[str, Any]) -> dash.Dash:
    """Create and configure the Dash application."""
    
    # Professional disclaimer text
    disclaimer_text = """
    Market Insight Platform - Advanced Analytics Dashboard. 
    This platform is designed for research and educational purposes only. 
    All data and analysis are for demonstration purposes. Not intended for live trading. 
    Past performance does not guarantee future results. Consult a financial advisor before making investment decisions.
    """
    
    # Initialize Dash app
    app = dash.Dash(
        __name__,
        title="Market Insight Platform | Professional Analytics",
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
        ],
        suppress_callback_exceptions=True
    )
    
    # Enhanced custom CSS with professional styling
    app.index_string = '''
    <!DOCTYPE html>
    <html lang="en">
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    background-attachment: fixed;
                    color: #202124;
                    line-height: 1.6;
                    -webkit-font-smoothing: antialiased;
                    -moz-osx-font-smoothing: grayscale;
                }
                
                /* Scrollbar styling */
                ::-webkit-scrollbar {
                    width: 10px;
                    height: 10px;
                }
                
                ::-webkit-scrollbar-track {
                    background: #f1f1f1;
                }
                
                ::-webkit-scrollbar-thumb {
                    background: #888;
                    border-radius: 5px;
                }
                
                ::-webkit-scrollbar-thumb:hover {
                    background: #555;
                }
                
                /* Loading spinner */
                ._dash-loading {
                    opacity: 0.3;
                }
                
                /* Card hover effects */
                .card-hover {
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                }
                
                .card-hover:hover {
                    transform: translateY(-4px);
                    box-shadow: 0 12px 24px rgba(0,0,0,0.12) !important;
                }
                
                /* Button styling */
                button {
                    transition: all 0.2s ease;
                }
                
                button:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                }
                
                button:active {
                    transform: translateY(0);
                }
                
                /* Dropdown styling */
                .Select-control {
                    border-radius: 8px !important;
                    border: 1px solid #e0e0e0 !important;
                    transition: all 0.2s ease !important;
                }
                
                .Select-control:hover {
                    border-color: #1e3c72 !important;
                }
                
                /* Tab styling */
                .tab {
                    transition: all 0.3s ease !important;
                }
                
                /* Chart container */
                .js-plotly-plot {
                    border-radius: 8px;
                }
                
                /* Positive/Negative indicators */
                .positive {
                    color: #0f9d58;
                    font-weight: 600;
                }
                
                .negative {
                    color: #db4437;
                    font-weight: 600;
                }
                
                .neutral {
                    color: #5f6368;
                    font-weight: 500;
                }
                
                /* KPI animations */
                @keyframes fadeInUp {
                    from {
                        opacity: 0;
                        transform: translateY(20px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }
                
                .animate-fade-in {
                    animation: fadeInUp 0.6s ease-out;
                }
                
                /* Grid layouts */
                .metric-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 16px;
                    margin: 16px 0;
                }
                
                /* Insight list styling */
                .insight-list {
                    list-style: none;
                    padding: 0;
                }
                
                .insight-list li {
                    padding: 12px 0;
                    border-bottom: 1px solid #e8eaed;
                    font-size: 14px;
                    color: #5f6368;
                    line-height: 1.6;
                }
                
                .insight-list li:last-child {
                    border-bottom: none;
                }
                
                .insight-list li::before {
                    content: "→";
                    color: #1e3c72;
                    font-weight: bold;
                    margin-right: 10px;
                }
                
                /* Metric card styling */
                .metric-card {
                    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                    border-radius: 8px;
                    padding: 16px;
                    border: 1px solid #e8eaed;
                    transition: all 0.3s ease;
                }
                
                .metric-card:hover {
                    border-color: #1e3c72;
                    box-shadow: 0 4px 12px rgba(30, 60, 114, 0.1);
                }
                
                .metric-title {
                    font-size: 12px;
                    font-weight: 600;
                    color: #5f6368;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    margin-bottom: 8px;
                }
                
                .metric-value {
                    font-size: 24px;
                    font-weight: 700;
                    color: #202124;
                    margin-bottom: 4px;
                }
                
                .metric-change {
                    font-size: 13px;
                    font-weight: 500;
                }
                
                /* DatePickerRange styling */
                .DateInput_input {
                    border-radius: 6px !important;
                    border: 1px solid #e0e0e0 !important;
                    padding: 8px 12px !important;
                    font-size: 14px !important;
                }
                
                .DateRangePickerInput {
                    border-radius: 8px !important;
                    border: 1px solid #e0e0e0 !important;
                }
                
                /* Radio items styling */
                .dash-radio-items label {
                    padding: 8px 0;
                    cursor: pointer;
                    transition: color 0.2s ease;
                }
                
                .dash-radio-items label:hover {
                    color: #1e3c72;
                }
                
                /* Responsive design */
                @media (max-width: 768px) {
                    .metric-grid {
                        grid-template-columns: 1fr;
                    }
                }
                
                /* Smooth scrolling */
                html {
                    scroll-behavior: smooth;
                }
                
                /* Focus accessibility */
                :focus {
                    outline: 2px solid #1e3c72;
                    outline-offset: 2px;
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
    try:
        app.layout = create_layout(config, disclaimer_text)
        logger.info("Dashboard layout created successfully")
    except Exception as e:
        logger.error(f"Error creating layout: {e}")
        raise
    
    # Register callbacks
    try:
        register_callbacks(app, config)
        logger.info("Dashboard callbacks registered successfully")
    except Exception as e:
        logger.error(f"Error registering callbacks: {e}")
        raise
    
    return app


if __name__ == "__main__":
    try:
        # Load configuration - SAME PATH AS ORIGINAL
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("Configuration loaded successfully")
        
        # Create and run app
        logger.info("Starting Market Insight Platform Dashboard...")
        app = create_app(config)
        
        # Run with same settings as original
        app.run(
            debug=False,
            host='127.0.0.1',
            port=8050,
            dev_tools_ui=False,
            dev_tools_props_check=False
        )
        
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        raise
