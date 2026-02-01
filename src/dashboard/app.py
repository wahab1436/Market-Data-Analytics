"""
Dashboard Application - Plotly Dash Interface
Professional market analytics dashboard with comprehensive data visualization
FIXED: Complete data loading and visualization for analysis and model results
"""

import dash
from dash import dcc, html, Input, Output, State
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import logging
import joblib
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataManager:
    """Manages loading and caching of analysis and model results."""
    
    def __init__(self, artifacts_path: str):
        self.artifacts_path = Path(artifacts_path)
        self._cache = {}
    
    def load_analysis_results(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Load analysis results for a symbol."""
        cache_key = f"analysis_{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        file_path = self.artifacts_path / f"{symbol}_analysis_results.pkl"
        try:
            if file_path.exists():
                data = joblib.load(file_path)
                self._cache[cache_key] = data
                logger.info(f"Loaded analysis results for {symbol}")
                return data
            else:
                logger.warning(f"Analysis file not found: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading analysis results for {symbol}: {e}")
            return None
    
    def load_model_results(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Load model results for a symbol."""
        cache_key = f"model_{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        file_path = self.artifacts_path / f"{symbol}_model_results.pkl"
        try:
            if file_path.exists():
                data = joblib.load(file_path)
                self._cache[cache_key] = data
                logger.info(f"Loaded model results for {symbol}")
                return data
            else:
                logger.warning(f"Model file not found: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading model results for {symbol}: {e}")
            return None
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from artifact files."""
        symbols = set()
        
        # Look for analysis files
        analysis_files = list(self.artifacts_path.glob("*_analysis_results.pkl"))
        for file_path in analysis_files:
            symbol = file_path.name.replace("_analysis_results.pkl", "")
            symbols.add(symbol)
        
        # Look for model files
        model_files = list(self.artifacts_path.glob("*_model_results.pkl"))
        for file_path in model_files:
            symbol = file_path.name.replace("_model_results.pkl", "")
            symbols.add(symbol)
        
        return sorted(list(symbols))


def create_figure_from_json(json_str: str, title: str = "") -> go.Figure:
    """Convert JSON string back to Plotly figure."""
    try:
        if isinstance(json_str, str) and json_str.strip():
            fig = pio.from_json(json_str)
            if title:
                fig.update_layout(title=title)
            return fig
        else:
            # Return empty figure if JSON is invalid
            fig = go.Figure()
            fig.add_annotation(text="No chart data available", x=0.5, y=0.5, showarrow=False)
            return fig
    except Exception as e:
        logger.error(f"Error creating figure from JSON: {e}")
        # Return empty figure with error message
        fig = go.Figure()
        fig.add_annotation(text=f"Error loading chart: {str(e)}", x=0.5, y=0.5, showarrow=False)
        return fig


def create_layout(config: Dict[str, Any], disclaimer_text: str, data_manager: DataManager) -> html.Div:
    """Create the main dashboard layout."""
    
    available_symbols = data_manager.get_available_symbols()
    default_symbol = available_symbols[0] if available_symbols else "AAPL"
    
    return html.Div([
        # Header
        html.Div([
            html.H1("📊 Market Insight Platform", 
                   style={'color': 'white', 'margin': '0', 'fontSize': '2.5rem'}),
            html.P("Professional Analytics Dashboard", 
                  style={'color': 'white', 'margin': '0', 'opacity': '0.9'})
        ], style={
            'background': 'linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)',
            'padding': '2rem',
            'textAlign': 'center',
            'borderBottom': '3px solid #ff6b6b'
        }),
        
        # Disclaimer
        html.Div([
            html.P(disclaimer_text, style={'margin': '0', 'fontSize': '0.9rem'})
        ], style={
            'background': '#fff3cd',
            'border': '1px solid #ffeaa7',
            'padding': '1rem',
            'margin': '1rem',
            'borderRadius': '8px',
            'color': '#856404'
        }),
        
        # Controls
        html.Div([
            html.Div([
                html.Label("Select Symbol:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Dropdown(
                    id='symbol-selector',
                    options=[{'label': sym, 'value': sym} for sym in available_symbols],
                    value=default_symbol,
                    style={'width': '200px', 'display': 'inline-block'}
                ),
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),
            
            html.Div([
                html.Label("Analysis Type:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.RadioItems(
                    id='analysis-type',
                    options=[
                        {'label': ' Price Analysis', 'value': 'price'},
                        {'label': ' Volatility Analysis', 'value': 'volatility'},
                        {'label': ' Volume Analysis', 'value': 'volume'},
                        {'label': ' Similarity Analysis', 'value': 'similarity'},
                        {'label': ' Regression Models', 'value': 'regression'},
                        {'label': ' KNN Similarity', 'value': 'knn'},
                        {'label': ' XGBoost Predictions', 'value': 'xgboost'}
                    ],
                    value='price',
                    inline=True,
                    labelStyle={'marginRight': '20px'}
                )
            ])
        ], style={
            'background': 'white',
            'padding': '20px',
            'margin': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
        }),
        
        # Content Area
        html.Div(id='content-area', style={'padding': '20px'}),
        
        # Footer
        html.Div([
            html.P("© 2024 Market Insight Platform | Built with Plotly Dash", 
                  style={'color': 'white', 'margin': '0', 'textAlign': 'center'})
        ], style={
            'background': '#2c3e50',
            'padding': '1rem',
            'marginTop': '2rem'
        })
    ], style={'minHeight': '100vh', 'fontFamily': 'Arial, sans-serif'})


def create_price_analysis_content(analysis_data: Dict[str, Any], symbol: str) -> html.Div:
    """Create price analysis content."""
    if not analysis_data or 'price' not in analysis_data:
        return html.Div("No price analysis data available")
    
    price_data = analysis_data['price']
    
    return html.Div([
        html.H2(f"Price Analysis - {symbol}", style={'color': '#1e3c72'}),
        
        # Metrics
        html.Div([
            html.Div([
                html.Div("Current Price", className="metric-title"),
                html.Div(f"${price_data.get('current_price', 'N/A'):.2f}", className="metric-value"),
            ], className="metric-card"),
            
            html.Div([
                html.Div("Daily Return", className="metric-title"),
                html.Div(f"{price_data.get('daily_return', 0)*100:.2f}%", 
                        className="metric-value positive" if price_data.get('daily_return', 0) > 0 else "metric-value negative"),
            ], className="metric-card"),
            
            html.Div([
                html.Div("Volatility", className="metric-title"),
                html.Div(f"{price_data.get('volatility', 0)*100:.2f}%", className="metric-value"),
            ], className="metric-card"),
        ], className="metric-grid"),
        
        # Charts
        html.Div([
            dcc.Graph(
                id='price-chart',
                figure=create_figure_from_json(
                    price_data.get('charts', {}).get('price_trend', ''),
                    f"{symbol} Price Trend"
                )
            )
        ], style={'marginTop': '20px'})
    ])


def create_model_content(model_data: Dict[str, Any], model_type: str, symbol: str) -> html.Div:
    """Create model analysis content."""
    if not model_data or model_type not in model_data:
        return html.Div(f"No {model_type} model data available")
    
    model_results = model_data[model_type]
    
    content = []
    content.append(html.H2(f"{model_type.upper()} Analysis - {symbol}", style={'color': '#1e3c72'}))
    
    # Metrics
    if 'metrics' in model_results:
        metrics_div = html.Div([], className="metric-grid")
        for key, value in model_results['metrics'].items():
            if isinstance(value, (int, float)):
                metric_card = html.Div([
                    html.Div(key.replace('_', ' ').title(), className="metric-title"),
                    html.Div(f"{value:.4f}", className="metric-value"),
                ], className="metric-card")
                metrics_div.children.append(metric_card)
        content.append(metrics_div)
    
    # Predictions chart
    if 'predictions' in model_results:
        # Create prediction chart from data
        predictions = model_results['predictions']
        if 'test' in predictions:
            test_data = predictions['test']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=test_data['dates'],
                y=test_data['actual'],
                name='Actual',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=test_data['dates'],
                y=test_data['predicted'],
                name='Predicted',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(title=f"{model_type.upper()} Predictions vs Actual")
            
            content.append(dcc.Graph(figure=fig))
    
    # Feature importance
    if 'feature_importance' in model_results:
        fi_data = model_results['feature_importance']
        if 'by_gain' in fi_data:
            features = [item['feature'] for item in fi_data['by_gain'][:10]]
            importance = [item.get('gain_pct', 0) for item in fi_data['by_gain'][:10]]
            
            fig = go.Figure(go.Bar(
                x=importance,
                y=features,
                orientation='h'
            ))
            fig.update_layout(title="Feature Importance")
            content.append(dcc.Graph(figure=fig))
    
    # Charts from JSON
    if 'charts' in model_results:
        for chart_name, chart_json in model_results['charts'].items():
            content.append(dcc.Graph(
                figure=create_figure_from_json(chart_json, f"{model_type} - {chart_name}")
            ))
    
    # Insights
    if 'insights' in model_results and model_results['insights']:
        insights_list = html.Ul([], className="insight-list")
        for insight in model_results['insights']:
            insights_list.children.append(html.Li(insight))
        content.append(html.Div([
            html.H3("Key Insights"),
            insights_list
        ]))
    
    return html.Div(content)


def register_callbacks(app: dash.Dash, config: Dict[str, Any], data_manager: DataManager):
    """Register all dashboard callbacks."""
    
    @app.callback(
        Output('content-area', 'children'),
        [Input('symbol-selector', 'value'),
         Input('analysis-type', 'value')]
    )
    def update_content(symbol, analysis_type):
        if not symbol:
            return html.Div("Please select a symbol")
        
        try:
            if analysis_type in ['price', 'volatility', 'volume', 'similarity']:
                # Analysis data
                analysis_data = data_manager.load_analysis_results(symbol)
                if not analysis_data:
                    return html.Div(f"No analysis data available for {symbol}")
                
                if analysis_type == 'price':
                    return create_price_analysis_content(analysis_data, symbol)
                elif analysis_type in analysis_data:
                    # For other analysis types
                    analysis_results = analysis_data[analysis_type]
                    
                    content = []
                    content.append(html.H2(f"{analysis_type.title()} Analysis - {symbol}"))
                    
                    # Metrics
                    if 'metrics' in analysis_results:
                        metrics_div = html.Div([], className="metric-grid")
                        for key, value in analysis_results['metrics'].items():
                            if isinstance(value, (int, float)):
                                metric_card = html.Div([
                                    html.Div(key.replace('_', ' ').title(), className="metric-title"),
                                    html.Div(f"{value:.4f}", className="metric-value"),
                                ], className="metric-card")
                                metrics_div.children.append(metric_card)
                        content.append(metrics_div)
                    
                    # Charts
                    if 'charts' in analysis_results:
                        for chart_name, chart_json in analysis_results['charts'].items():
                            content.append(dcc.Graph(
                                figure=create_figure_from_json(chart_json, f"{analysis_type} - {chart_name}")
                            ))
                    
                    # Insights
                    if 'insights' in analysis_results and analysis_results['insights']:
                        insights_list = html.Ul([], className="insight-list")
                        for insight in analysis_results['insights']:
                            insights_list.children.append(html.Li(insight))
                        content.append(html.Div([
                            html.H3("Key Insights"),
                            insights_list
                        ]))
                    
                    return html.Div(content)
                else:
                    return html.Div(f"No {analysis_type} analysis available for {symbol}")
            
            elif analysis_type in ['regression', 'knn', 'xgboost']:
                # Model data
                model_data = data_manager.load_model_results(symbol)
                return create_model_content(model_data, analysis_type, symbol)
            
            else:
                return html.Div("Invalid analysis type")
                
        except Exception as e:
            logger.error(f"Error updating content: {e}")
            return html.Div([
                html.H3("Error Loading Data"),
                html.P(str(e)),
                html.P("Please check if data files exist and try running the batch analysis first.")
            ])


def create_app(config: Dict[str, Any]) -> dash.Dash:
    """Create and configure the Dash application."""
    
    # Initialize data manager
    artifacts_path = config['paths']['artifacts']
    data_manager = DataManager(artifacts_path)
    
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
                
                /* Metric card styling */
                .metric-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 16px;
                    margin: 20px 0;
                }
                
                .metric-card {
                    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                    border-radius: 8px;
                    padding: 20px;
                    border: 1px solid #e8eaed;
                    transition: all 0.3s ease;
                    text-align: center;
                }
                
                .metric-card:hover {
                    border-color: #1e3c72;
                    box-shadow: 0 4px 12px rgba(30, 60, 114, 0.1);
                    transform: translateY(-2px);
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
                
                .positive {
                    color: #0f9d58;
                }
                
                .negative {
                    color: #db4437;
                }
                
                /* Insight list styling */
                .insight-list {
                    list-style: none;
                    padding: 0;
                    background: white;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px 0;
                    border: 1px solid #e8eaed;
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
                
                /* Responsive design */
                @media (max-width: 768px) {
                    .metric-grid {
                        grid-template-columns: 1fr;
                    }
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
    app.layout = create_layout(config, disclaimer_text, data_manager)
    logger.info("Dashboard layout created successfully")
    
    # Register callbacks
    register_callbacks(app, config, data_manager)
    logger.info("Dashboard callbacks registered successfully")
    
    return app


if __name__ == "__main__":
    try:
        # Load configuration
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        
        if not config_path.exists():
            # Try alternative path
            config_path = Path(__file__).parent / "config.yaml"
            
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from: {config_path}")
        logger.info(f"Artifacts path: {config['paths']['artifacts']}")
        
        # Check if artifacts directory exists
        artifacts_path = Path(config['paths']['artifacts'])
        if not artifacts_path.exists():
            logger.warning(f"Artifacts directory does not exist: {artifacts_path}")
            logger.info("Please run 'python main.py --mode batch' to generate data first")
        else:
            # Check what files are available
            analysis_files = list(artifacts_path.glob("*_analysis_results.pkl"))
            model_files = list(artifacts_path.glob("*_model_results.pkl"))
            logger.info(f"Found {len(analysis_files)} analysis files and {len(model_files)} model files")
        
        # Create and run app
        logger.info("Starting Market Insight Platform Dashboard...")
        app = create_app(config)
        
        # Run with debugging enabled
        app.run(
            debug=True,
            host='127.0.0.1',
            port=8050,
            dev_tools_ui=True,
            dev_tools_props_check=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        raise
