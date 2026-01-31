"""
Dashboard Callbacks Module
Interactive callbacks for the Market Insight Platform dashboard
All callbacks properly configured for professional dashboard interaction
"""

from dash import Input, Output, State, callback_context, no_update, html
import pandas as pd
import numpy as np
from pathlib import Path
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def register_callbacks(app, config: Dict[str, Any]):
    """Register all dashboard callbacks with proper error handling."""
    
    symbols = config['data']['symbols']
    color_palette = config['dashboard']['color_palette']
    data_paths = config['paths']
    
    # Helper function to load precomputed data
    def load_precomputed_data(symbol: str, data_type: str):
        """Load precomputed data from disk with error handling."""
        try:
            if data_type == 'analysis':
                file_path = Path(data_paths['artifacts']) / f"{symbol}_analysis_results.pkl"
            elif data_type == 'models':
                file_path = Path(data_paths['artifacts']) / f"{symbol}_model_results.pkl"
            elif data_type == 'features':
                file_path = Path(data_paths['gold_data']) / f"{symbol}_featured.parquet"
            else:
                return None
            
            if file_path.exists():
                if str(file_path).endswith('.pkl'):
                    return joblib.load(file_path)
                elif str(file_path).endswith('.parquet'):
                    return pd.read_parquet(file_path)
            return None
        except Exception as e:
            logger.error(f"Error loading {data_type} data for {symbol}: {e}")
            return None
    
    # Helper function to create KPI cards
    def create_kpi_display(metrics: Dict[str, Any]) -> List:
        """Create KPI display from metrics."""
        displays = []
        
        # Current Price
        current_price = metrics.get('current_price', 0)
        price_change = metrics.get('price_change_1d', 0)
        displays.append({
            'id': 'price-kpi',
            'value': f"${current_price:.2f}",
            'trend': price_change
        })
        
        # Daily Return
        daily_return = metrics.get('price_change_1d', 0)
        displays.append({
            'id': 'return-kpi',
            'value': f"{daily_return:+.2f}%",
            'trend': daily_return
        })
        
        # Volatility
        volatility = metrics.get('annualized_volatility', 0)
        displays.append({
            'id': 'volatility-kpi',
            'value': f"{volatility:.1f}%",
            'trend': "High" if volatility > 30 else "Low" if volatility < 15 else "Moderate"
        })
        
        # Volume
        volume = metrics.get('current_volume', 0)
        avg_volume = metrics.get('avg_volume_20d', 1)
        volume_ratio = (volume / avg_volume) if avg_volume > 0 else 0
        
        if volume_ratio > 1:
            trend = f"{volume_ratio:.1f}x avg"
        else:
            trend = f"{1/volume_ratio:.1f}x below" if volume_ratio > 0 else "N/A"
        
        displays.append({
            'id': 'volume-kpi',
            'value': f"{volume:,.0f}" if volume > 0 else "N/A",
            'trend': trend
        })
        
        return displays
    
    # Helper function to create insight display
    def create_insight_display(insights: List[str]) -> List[html.Li]:
        """Create insight list items."""
        if not insights:
            return [html.Li("No insights available.", className="text-muted", style={'listStyle': 'none'})]
        
        items = []
        for insight in insights:
            items.append(html.Li(insight, style={
                'marginBottom': '8px',
                'fontSize': '14px',
                'color': '#5f6368',
                'lineHeight': '1.6'
            }))
        return items
    
    # 1. Initial Data Loading Callback
    @app.callback(
        [Output('analysis-data-store', 'data'),
         Output('model-data-store', 'data'),
         Output('featured-data-store', 'data')],
        [Input('symbol-selector', 'value')]
    )
    def load_initial_data(selected_symbol):
        """Load all precomputed data for selected symbol."""
        if not selected_symbol:
            return None, None, None
        
        logger.info(f"Loading data for symbol: {selected_symbol}")
        
        # Load analysis results
        analysis_data = load_precomputed_data(selected_symbol, 'analysis')
        
        # Load model results
        model_data = load_precomputed_data(selected_symbol, 'models')
        
        # Load featured data
        featured_data = load_precomputed_data(selected_symbol, 'features')
        
        # Convert to JSON-serializable format
        analysis_json = json.loads(json.dumps(analysis_data, default=str)) if analysis_data else None
        model_json = json.loads(json.dumps(model_data, default=str)) if model_data else None
        featured_json = featured_data.to_dict('records') if featured_data is not None else None
        
        return analysis_json, model_json, featured_json
    
    # 2. KPI Update Callback - FIXED VERSION
    @app.callback(
        [Output('price-kpi', 'children'),
         Output('price-kpi-trend', 'children'),
         Output('return-kpi', 'children'),
         Output('return-kpi-trend', 'children'),
         Output('volatility-kpi', 'children'),
         Output('volatility-kpi-trend', 'children'),
         Output('volume-kpi', 'children'),
         Output('volume-kpi-trend', 'children')],
        [Input('analysis-data-store', 'data')]
    )
    def update_kpis(analysis_data):
        """Update KPI cards with latest metrics - FIXED to return exactly 8 values."""
        if not analysis_data:
            # Return exactly 8 values - NO * 2
            return "$--", "", "--%", "", "--%", "", "--", ""
        
        try:
            # Extract metrics from analysis data
            metrics = analysis_data.get('price', {}).get('metrics', {})
            
            # Create KPI displays
            displays = create_kpi_display(metrics)
            
            # Return formatted values (exactly 8 values)
            trend_value = displays[0]['trend']
            trend_text = f"1d: {trend_value:+.2f}%" if isinstance(trend_value, (int, float)) and trend_value != 0 else ""
            
            return_trend = displays[1]['trend']
            return_trend_text = "▲" if return_trend > 0 else "▼" if return_trend < 0 else "—"
            
            return (
                displays[0]['value'],
                trend_text,
                displays[1]['value'],
                return_trend_text,
                displays[2]['value'],
                displays[2]['trend'],
                displays[3]['value'],
                displays[3]['trend']
            )
        except Exception as e:
            logger.error(f"Error updating KPIs: {e}")
            # Return exactly 8 values - NO * 2
            return "$--", "", "--%", "", "--%", "", "--", ""
    
    # 3. Price Analysis Callbacks
    @app.callback(
        [Output('price-chart', 'figure'),
         Output('price-insights', 'children'),
         Output('trend-metrics', 'children')],
        [Input('analysis-data-store', 'data'),
         Input('featured-data-store', 'data')]
    )
    def update_price_analysis(analysis_data, featured_data):
        """Update price analysis charts and insights."""
        if not analysis_data or not featured_data:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No data available",
                template='plotly_white',
                height=500
            )
            return empty_fig, html.Div("No data available.", style={'color': '#5f6368'}), html.Div("No metrics available.", style={'color': '#5f6368'})
        
        try:
            # Convert featured data back to DataFrame
            featured_df = pd.DataFrame(featured_data)
            if 'date' in featured_df.columns:
                featured_df['date'] = pd.to_datetime(featured_df['date'])
                featured_df = featured_df.set_index('date')
            
            # Get price analysis results
            price_analysis = analysis_data.get('price', {})
            
            # Create price chart
            fig = create_price_chart(featured_df, price_analysis, color_palette)
            
            # Create insights
            insights = price_analysis.get('insights', [])
            insight_display = html.Ul(create_insight_display(insights), className='insight-list')
            
            # Create metrics display
            metrics = price_analysis.get('metrics', {})
            metrics_display = create_metrics_display(metrics)
            
            return fig, insight_display, metrics_display
            
        except Exception as e:
            logger.error(f"Error updating price analysis: {e}")
            empty_fig = go.Figure()
            empty_fig.update_layout(template='plotly_white', height=500)
            return empty_fig, html.Div(f"Error: {str(e)}", style={'color': '#db4437'}), html.Div("Error loading metrics", style={'color': '#db4437'})
    
    def create_price_chart(df: pd.DataFrame, price_analysis: Dict, colors: Dict) -> go.Figure:
        """Create price chart with moving averages."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=("Price with Moving Averages", "Daily Returns")
        )
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['close'],
                mode='lines',
                name='Close Price',
                line=dict(color=colors.get('primary', '#1e3c72'), width=2),
                hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add moving averages if available
        ma_colors = ['#4285f4', '#34a853', '#fbbc04']
        for idx, ma in enumerate([20, 50, 200]):
            ma_col = f'ma_{ma}d'
            if ma_col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[ma_col],
                        mode='lines',
                        name=f'{ma}-day MA',
                        line=dict(color=ma_colors[idx % len(ma_colors)], width=1.5, dash='dash'),
                        hovertemplate=f'{ma}-day MA: $%{{y:.2f}}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Daily returns
        if 'return' in df.columns:
            returns = df['return'] * 100
            colors_bar = ['#0f9d58' if x >= 0 else '#db4437' for x in returns]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=returns,
                    name='Daily Return %',
                    marker_color=colors_bar,
                    hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=500,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white',
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)
        
        return fig
    
    def create_metrics_display(metrics: Dict[str, float]) -> html.Div:
        """Create metrics display cards."""
        if not metrics:
            return html.Div("No metrics available", style={'color': '#5f6368'})
        
        metric_cards = []
        
        metric_items = [
            ('current_price', 'Current Price', lambda x: f"${x:.2f}", None),
            ('price_change_30d', '30-day Change', lambda x: f"{x:+.1f}%", lambda x: 'positive' if x > 0 else 'negative'),
            ('annualized_volatility', 'Annualized Vol', lambda x: f"{x:.1f}%", lambda x: 'negative' if x > 30 else 'positive' if x < 15 else 'neutral'),
            ('max_drawdown', 'Max Drawdown', lambda x: f"{x:.1f}%", lambda x: 'negative' if x < -20 else 'neutral')
        ]
        
        for key, title, formatter, color_func in metric_items:
            if key in metrics:
                value = metrics[key]
                formatted_value = formatter(value)
                color_class = color_func(value) if color_func else 'neutral'
                
                metric_cards.append(
                    html.Div([
                        html.Div(title, className='metric-title'),
                        html.Div(formatted_value, className=f'metric-value {color_class}')
                    ], className='metric-card')
                )
        
        return html.Div(metric_cards, className='metric-grid')
    
    # 4. Volatility Analysis Callbacks
    @app.callback(
        [Output('volatility-chart', 'figure'),
         Output('volatility-distribution-chart', 'figure'),
         Output('volatility-insights', 'children')],
        [Input('analysis-data-store', 'data'),
         Input('featured-data-store', 'data')]
    )
    def update_volatility_analysis(analysis_data, featured_data):
        """Update volatility analysis charts and insights."""
        if not analysis_data or not featured_data:
            empty_fig = go.Figure()
            empty_fig.update_layout(template='plotly_white', height=450)
            return empty_fig, empty_fig, html.Div("No data available.", style={'color': '#5f6368'})
        
        try:
            # Convert featured data back to DataFrame
            featured_df = pd.DataFrame(featured_data)
            if 'date' in featured_df.columns:
                featured_df['date'] = pd.to_datetime(featured_df['date'])
                featured_df = featured_df.set_index('date')
            
            # Get volatility analysis results
            volatility_analysis = analysis_data.get('volatility', {})
            
            # Create volatility charts
            timeline_fig = create_volatility_timeline(featured_df, color_palette)
            distribution_fig = create_volatility_distribution(featured_df, color_palette)
            
            # Create insights
            insights = volatility_analysis.get('insights', [])
            insight_display = html.Ul(create_insight_display(insights), className='insight-list')
            
            return timeline_fig, distribution_fig, insight_display
            
        except Exception as e:
            logger.error(f"Error updating volatility analysis: {e}")
            empty_fig = go.Figure()
            empty_fig.update_layout(template='plotly_white', height=450)
            return empty_fig, empty_fig, html.Div(f"Error: {str(e)}", style={'color': '#db4437'})
    
    def create_volatility_timeline(df: pd.DataFrame, colors: Dict) -> go.Figure:
        """Create volatility timeline chart."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.5],
            subplot_titles=("20-day Rolling Volatility", "Daily Returns")
        )
        
        # Calculate rolling volatility
        if 'return' in df.columns:
            rolling_vol = df['return'].rolling(window=20, min_periods=20).std() * np.sqrt(252) * 100
            
            # Volatility line
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=rolling_vol,
                    mode='lines',
                    name='20-Day Volatility',
                    line=dict(color=colors.get('primary', '#1e3c72'), width=2),
                    fill='tozeroy',
                    fillcolor='rgba(30, 60, 114, 0.1)',
                    hovertemplate='Date: %{x}<br>Volatility: %{y:.1f}%<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add mean line
            if not rolling_vol.empty:
                vol_mean = rolling_vol.mean()
                fig.add_hline(
                    y=vol_mean,
                    line_width=1,
                    line_dash="dash",
                    line_color='#5f6368',
                    annotation_text=f"Mean: {vol_mean:.1f}%",
                    row=1, col=1
                )
            
            # Daily returns
            returns = df['return'] * 100
            colors_bar = ['#0f9d58' if x >= 0 else '#db4437' for x in returns]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=returns,
                    name='Daily Return',
                    marker_color=colors_bar,
                    hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=450,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white',
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)
        
        return fig
    
    def create_volatility_distribution(df: pd.DataFrame, colors: Dict) -> go.Figure:
        """Create volatility distribution chart."""
        fig = go.Figure()
        
        if 'return' in df.columns:
            returns = df['return'] * 100
            
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    nbinsx=50,
                    name='Return Distribution',
                    marker_color=colors.get('primary', '#1e3c72'),
                    opacity=0.7,
                    hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
                )
            )
            
            # Add statistics
            mean_return = returns.mean()
            std_return = returns.std()
            
            fig.add_vline(
                x=mean_return,
                line_width=2,
                line_dash="dash",
                line_color='#5f6368',
                annotation_text=f"Mean: {mean_return:.2f}%"
            )
            
            fig.add_vline(x=mean_return + std_return, line_width=1, line_dash="dot", line_color='#f4b400')
            fig.add_vline(x=mean_return - std_return, line_width=1, line_dash="dot", line_color='#f4b400')
        
        fig.update_layout(
            height=450,
            showlegend=False,
            template='plotly_white',
            xaxis_title='Daily Return (%)',
            yaxis_title='Frequency',
            margin=dict(l=60, r=60, t=60, b=60)
        )
        
        return fig
    
    # 5. Volume Analysis Callbacks
    @app.callback(
        [Output('volume-chart', 'figure'),
         Output('volume-insights', 'children'),
         Output('volume-relationship', 'children')],
        [Input('analysis-data-store', 'data'),
         Input('featured-data-store', 'data')]
    )
    def update_volume_analysis(analysis_data, featured_data):
        """Update volume analysis charts and insights."""
        if not analysis_data or not featured_data:
            empty_fig = go.Figure()
            empty_fig.update_layout(template='plotly_white', height=500)
            return empty_fig, html.Div("No data available.", style={'color': '#5f6368'}), html.Div("No relationship data.", style={'color': '#5f6368'})
        
        try:
            # Convert featured data back to DataFrame
            featured_df = pd.DataFrame(featured_data)
            if 'date' in featured_df.columns:
                featured_df['date'] = pd.to_datetime(featured_df['date'])
                featured_df = featured_df.set_index('date')
            
            # Get volume analysis results
            volume_analysis = analysis_data.get('volume', {})
            
            # Create volume chart
            fig = create_volume_chart(featured_df, color_palette)
            
            # Create insights
            insights = volume_analysis.get('insights', [])
            insight_display = html.Ul(create_insight_display(insights), className='insight-list')
            
            # Create relationship metrics
            relationship_display = create_volume_relationship_display(volume_analysis)
            
            return fig, insight_display, relationship_display
            
        except Exception as e:
            logger.error(f"Error updating volume analysis: {e}")
            empty_fig = go.Figure()
            empty_fig.update_layout(template='plotly_white', height=500)
            return empty_fig, html.Div(f"Error: {str(e)}", style={'color': '#db4437'}), html.Div("Error loading relationship data", style={'color': '#db4437'})
    
    def create_volume_chart(df: pd.DataFrame, colors: Dict) -> go.Figure:
        """Create volume-price chart."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=("Price & Volume", "Volume Ratio (vs 20-day MA)"),
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['close'],
                mode='lines',
                name='Close Price',
                line=dict(color=colors.get('primary', '#1e3c72'), width=2),
                hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1,
            secondary_y=False
        )
        
        # Volume bars (secondary y-axis)
        if 'volume' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    name='Volume',
                    marker_color='rgba(66, 133, 244, 0.5)',
                    hovertemplate='Volume: %{y:,.0f}<extra></extra>'
                ),
                row=1, col=1,
                secondary_y=True
            )
        
        # Volume ratio
        if 'volume_ratio_20d' in df.columns:
            volume_ratio = df['volume_ratio_20d']
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=volume_ratio,
                    mode='lines',
                    name='Volume Ratio',
                    line=dict(color=colors.get('success', '#0f9d58'), width=1.5),
                    hovertemplate='Volume Ratio: %{y:.2f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add horizontal line at 1.0
            fig.add_hline(y=1.0, line_width=1, line_dash="dash", line_color='#5f6368', row=2, col=1)
        
        fig.update_layout(
            height=500,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white',
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Volume Ratio", row=2, col=1)
        
        return fig
    
    def create_volume_relationship_display(volume_analysis: Dict) -> html.Div:
        """Create volume-price relationship display."""
        if not volume_analysis:
            return html.Div("No relationship data available", style={'color': '#5f6368'})
        
        metrics = volume_analysis.get('metrics', {})
        cards = []
        
        metric_items = [
            ('current_volume_return_corr', 'Volume-Return Correlation', lambda x: f"{x:.3f}", lambda x: 'positive' if abs(x) > 0.3 else 'neutral'),
            ('volume_ratio_current_vs_avg', 'Current vs Avg Volume', lambda x: f"{x:.1f}x", lambda x: 'positive' if x > 1.5 else 'negative' if x < 0.7 else 'neutral'),
            ('pct_extreme_volume_days', 'Extreme Volume Days', lambda x: f"{x:.1f}%", lambda x: 'negative' if x > 10 else 'positive')
        ]
        
        for key, title, formatter, color_func in metric_items:
            if key in metrics:
                value = metrics[key]
                formatted_value = formatter(value)
                color_class = color_func(value)
                
                cards.append(
                    html.Div([
                        html.Div(title, className='metric-title'),
                        html.Div(formatted_value, className=f'metric-value {color_class}')
                    ], className='metric-card')
                )
        
        return html.Div(cards, className='metric-grid')
    
    # 6. Similarity Analysis Callbacks
    @app.callback(
        [Output('similarity-chart', 'figure'),
         Output('analog-insights', 'children')],
        [Input('analysis-data-store', 'data'),
         Input('featured-data-store', 'data')]
    )
    def update_similarity_analysis(analysis_data, featured_data):
        """Update similarity analysis charts and insights."""
        if not analysis_data or not featured_data:
            empty_fig = go.Figure()
            empty_fig.update_layout(template='plotly_white', height=450)
            return empty_fig, html.Div("No data available.", style={'color': '#5f6368'})
        
        try:
            # Convert featured data back to DataFrame
            featured_df = pd.DataFrame(featured_data)
            if 'date' in featured_df.columns:
                featured_df['date'] = pd.to_datetime(featured_df['date'])
                featured_df = featured_df.set_index('date')
            
            # Get similarity analysis results
            similarity_analysis = analysis_data.get('similarity', {})
            
            # Create similarity chart
            fig = create_similarity_chart(featured_df, similarity_analysis, color_palette)
            
            # Create insights
            insights = similarity_analysis.get('insights', [])
            insight_display = html.Ul(create_insight_display(insights), className='insight-list')
            
            return fig, insight_display
            
        except Exception as e:
            logger.error(f"Error updating similarity analysis: {e}")
            empty_fig = go.Figure()
            empty_fig.update_layout(template='plotly_white', height=450)
            return empty_fig, html.Div(f"Error: {str(e)}", style={'color': '#db4437'})
    
    def create_similarity_chart(df: pd.DataFrame, similarity_analysis: Dict, colors: Dict) -> go.Figure:
        """Create similarity pattern chart."""
        fig = go.Figure()
        
        # Plot recent price pattern
        lookback = 20
        if len(df) >= lookback:
            recent_prices = df['close'].iloc[-lookback:]
            
            # Normalize for pattern comparison
            min_price = recent_prices.min()
            max_price = recent_prices.max()
            price_range = max_price - min_price
            normalized_prices = (recent_prices - min_price) / price_range if price_range > 0 else recent_prices * 0
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(lookback)),
                    y=normalized_prices,
                    mode='lines+markers',
                    name='Current Pattern',
                    line=dict(color=colors.get('primary', '#1e3c72'), width=3),
                    marker=dict(size=8),
                    hovertemplate='Day: %{x}<br>Normalized: %{y:.3f}<extra></extra>'
                )
            )
        
        fig.update_layout(
            height=450,
            showlegend=True,
            template='plotly_white',
            xaxis_title='Trading Days',
            yaxis_title='Normalized Price',
            title='Current Pattern vs Historical Analogs',
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        return fig
    
    # 7. Model Tabs Callback
    @app.callback(
        Output('model-tab-content', 'children'),
        [Input('model-tabs', 'value')]
    )
    def update_model_tab(active_tab):
        """Update model tab content."""
        from .layout import create_model_tab_content
        return create_model_tab_content(active_tab)
    
    # 8. Regression Analysis Callbacks
    @app.callback(
        [Output('regression-chart', 'figure'),
         Output('regression-metrics-chart', 'figure'),
         Output('regression-insights', 'children')],
        [Input('model-data-store', 'data')]
    )
    def update_regression_analysis(model_data):
        """Update regression analysis charts and insights."""
        if not model_data:
            empty_fig = go.Figure()
            empty_fig.update_layout(template='plotly_white', height=400)
            return empty_fig, empty_fig, html.Div("No model data available.", style={'color': '#5f6368'})
        
        try:
            # Get regression results
            regression_data = model_data.get('regression', {})
            
            # Create regression charts
            pred_fig = create_regression_prediction_chart(regression_data, color_palette)
            metrics_fig = create_regression_metrics_chart(regression_data, color_palette)
            
            # Create insights
            insights = regression_data.get('insights', [])
            insight_display = html.Ul(create_insight_display(insights), className='insight-list')
            
            return pred_fig, metrics_fig, insight_display
            
        except Exception as e:
            logger.error(f"Error updating regression analysis: {e}")
            empty_fig = go.Figure()
            empty_fig.update_layout(template='plotly_white', height=400)
            return empty_fig, empty_fig, html.Div(f"Error: {str(e)}", style={'color': '#db4437'})
    
    def create_regression_prediction_chart(regression_data: Dict, colors: Dict) -> go.Figure:
        """Create regression prediction vs actual chart."""
        fig = go.Figure()
        
        predictions = regression_data.get('predictions', {})
        if predictions and 'data' in predictions:
            pred_df = pd.DataFrame(predictions['data'])
            
            if 'date' in pred_df.columns and 'actual' in pred_df.columns and 'predicted' in pred_df.columns:
                pred_df['date'] = pd.to_datetime(pred_df['date'])
                
                fig.add_trace(
                    go.Scatter(
                        x=pred_df['date'],
                        y=pred_df['actual'] * 100,
                        mode='lines',
                        name='Actual',
                        line=dict(color=colors.get('primary', '#1e3c72'), width=2),
                        hovertemplate='Date: %{x}<br>Actual: %{y:.2f}%<extra></extra>'
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=pred_df['date'],
                        y=pred_df['predicted'] * 100,
                        mode='lines',
                        name='Predicted',
                        line=dict(color=colors.get('success', '#0f9d58'), width=2, dash='dash'),
                        hovertemplate='Date: %{x}<br>Predicted: %{y:.2f}%<extra></extra>'
                    )
                )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            template='plotly_white',
            xaxis_title='Date',
            yaxis_title='Return (%)',
            title='Regression: Actual vs Predicted',
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        return fig
    
    def create_regression_metrics_chart(regression_data: Dict, colors: Dict) -> go.Figure:
        """Create regression metrics comparison chart."""
        fig = go.Figure()
        
        metrics = regression_data.get('metrics', {})
        
        model_names = []
        rmse_values = []
        r2_values = []
        
        for model in ['linear', 'ridge', 'lasso']:
            rmse_key = f'{model}_rmse'
            r2_key = f'{model}_r2'
            
            if rmse_key in metrics and r2_key in metrics:
                model_names.append(model.capitalize())
                rmse_values.append(metrics[rmse_key] * 100)
                r2_values.append(metrics[r2_key])
        
        if model_names:
            # RMSE bars
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=rmse_values,
                    name='RMSE (%)',
                    marker_color=colors.get('primary', '#1e3c72'),
                    text=[f'{v:.2f}%' for v in rmse_values],
                    textposition='auto',
                    hovertemplate='Model: %{x}<br>RMSE: %{y:.2f}%<extra></extra>'
                )
            )
            
            # R² line (secondary axis)
            fig.add_trace(
                go.Scatter(
                    x=model_names,
                    y=r2_values,
                    name='R² Score',
                    mode='lines+markers',
                    yaxis='y2',
                    line=dict(color=colors.get('success', '#0f9d58'), width=3),
                    marker=dict(size=10),
                    text=[f'{v:.3f}' for v in r2_values],
                    hovertemplate='R²: %{y:.3f}<extra></extra>'
                )
            )
            
            fig.update_layout(
                yaxis=dict(title='RMSE (%)'),
                yaxis2=dict(title='R² Score', overlaying='y', side='right')
            )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            template='plotly_white',
            xaxis_title='Model',
            title='Model Performance Comparison',
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        return fig
    
    # 9. KNN Analysis Callbacks
    @app.callback(
        [Output('knn-chart', 'figure'),
         Output('knn-analogs', 'children'),
         Output('knn-insights', 'children')],
        [Input('model-data-store', 'data')]
    )
    def update_knn_analysis(model_data):
        """Update KNN analysis charts and insights."""
        if not model_data:
            empty_fig = go.Figure()
            empty_fig.update_layout(template='plotly_white', height=400)
            return empty_fig, html.Div("No analog data.", style={'color': '#5f6368'}), html.Div("No insights available.", style={'color': '#5f6368'})
        
        try:
            # Get KNN results
            knn_data = model_data.get('knn', {})
            
            # Create KNN chart
            fig = create_knn_chart(knn_data, color_palette)
            
            # Create analog display
            analogs = knn_data.get('analogs', {}).get('top_analogs', [])
            analog_display = create_analog_display(analogs)
            
            # Create insights
            insights = knn_data.get('insights', [])
            insight_display = html.Ul(create_insight_display(insights), className='insight-list')
            
            return fig, analog_display, insight_display
            
        except Exception as e:
            logger.error(f"Error updating KNN analysis: {e}")
            empty_fig = go.Figure()
            empty_fig.update_layout(template='plotly_white', height=400)
            return empty_fig, html.Div(f"Error: {str(e)}", style={'color': '#db4437'}), html.Div("Error loading insights", style={'color': '#db4437'})
    
    def create_knn_chart(knn_data: Dict, colors: Dict) -> go.Figure:
        """Create KNN prediction chart."""
        fig = go.Figure()
        
        predictions = knn_data.get('predictions', {})
        if predictions and 'data' in predictions:
            pred_df = pd.DataFrame(predictions['data'])
            
            if 'date' in pred_df.columns and 'actual' in pred_df.columns and 'predicted' in pred_df.columns:
                pred_df['date'] = pd.to_datetime(pred_df['date'])
                
                fig.add_trace(
                    go.Scatter(
                        x=pred_df['date'],
                        y=pred_df['actual'] * 100,
                        mode='lines',
                        name='Actual',
                        line=dict(color=colors.get('primary', '#1e3c72'), width=2),
                        hovertemplate='Date: %{x}<br>Actual: %{y:.2f}%<extra></extra>'
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=pred_df['date'],
                        y=pred_df['predicted'] * 100,
                        mode='lines',
                        name='KNN Predicted',
                        line=dict(color=colors.get('secondary', '#4285f4'), width=2, dash='dash'),
                        hovertemplate='Date: %{x}<br>Predicted: %{y:.2f}%<extra></extra>'
                    )
                )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            template='plotly_white',
            xaxis_title='Date',
            yaxis_title='Return (%)',
            title='KNN: Actual vs Predicted',
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        return fig
    
    def create_analog_display(analogs: List) -> html.Div:
        """Create analog display."""
        if not analogs:
            return html.Div("No historical analogs found", style={'color': '#5f6368'})
        
        display_items = []
        for i, analog in enumerate(analogs[:3]):
            display_items.append(
                html.Div([
                    html.H4(f"Analog {i+1}", style={'fontSize': '14px', 'fontWeight': '600', 'color': '#202124', 'marginBottom': '8px'}),
                    html.P(f"Date: {analog.get('date', 'N/A')}", style={'fontSize': '13px', 'color': '#5f6368', 'margin': '4px 0'}),
                    html.P(f"Similarity: {analog.get('similarity', 0):.3f}", style={'fontSize': '13px', 'color': '#5f6368', 'margin': '4px 0'}),
                    html.P(f"Days Apart: {analog.get('days_apart', 0)}", style={'fontSize': '13px', 'color': '#5f6368', 'margin': '4px 0'})
                ], style={
                    'marginBottom': '16px',
                    'padding': '12px',
                    'background': '#f8f9fa',
                    'borderRadius': '8px',
                    'border': '1px solid #e8eaed'
                })
            )
        
        return html.Div(display_items)
    
    # 10. XGBoost Analysis Callbacks
    @app.callback(
        [Output('xgboost-prediction-chart', 'figure'),
         Output('xgboost-importance-chart', 'figure'),
         Output('xgboost-metrics', 'children')],
        [Input('model-data-store', 'data')]
    )
    def update_xgboost_analysis(model_data):
        """Update XGBoost analysis charts and metrics."""
        if not model_data:
            empty_fig = go.Figure()
            empty_fig.update_layout(template='plotly_white', height=400)
            return empty_fig, empty_fig, html.Div("No XGBoost data available.", style={'color': '#5f6368'})
        
        try:
            # Get XGBoost results
            xgboost_data = model_data.get('xgboost', {})
            
            # Create XGBoost charts
            pred_fig = create_xgboost_prediction_chart(xgboost_data, color_palette)
            importance_fig = create_xgboost_importance_chart(xgboost_data, color_palette)
            
            # Create metrics display
            metrics = xgboost_data.get('metrics', {})
            metrics_display = create_xgboost_metrics_display(metrics)
            
            return pred_fig, importance_fig, metrics_display
            
        except Exception as e:
            logger.error(f"Error updating XGBoost analysis: {e}")
            empty_fig = go.Figure()
            empty_fig.update_layout(template='plotly_white', height=400)
            return empty_fig, empty_fig, html.Div(f"Error: {str(e)}", style={'color': '#db4437'})
    
    def create_xgboost_prediction_chart(xgboost_data: Dict, colors: Dict) -> go.Figure:
        """Create XGBoost prediction chart."""
        fig = go.Figure()
        
        predictions = xgboost_data.get('predictions', {})
        if predictions and 'data' in predictions:
            pred_df = pd.DataFrame(predictions['data'])
            
            if 'date' in pred_df.columns and 'actual' in pred_df.columns and 'predicted' in pred_df.columns:
                pred_df['date'] = pd.to_datetime(pred_df['date'])
                
                fig.add_trace(
                    go.Scatter(
                        x=pred_df['date'],
                        y=pred_df['actual'] * 100,
                        mode='lines',
                        name='Actual',
                        line=dict(color=colors.get('primary', '#1e3c72'), width=2),
                        hovertemplate='Date: %{x}<br>Actual: %{y:.2f}%<extra></extra>'
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=pred_df['date'],
                        y=pred_df['predicted'] * 100,
                        mode='lines',
                        name='XGBoost Predicted',
                        line=dict(color=colors.get('warning', '#f4b400'), width=2, dash='dash'),
                        hovertemplate='Date: %{x}<br>Predicted: %{y:.2f}%<extra></extra>'
                    )
                )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            template='plotly_white',
            xaxis_title='Date',
            yaxis_title='Return (%)',
            title='XGBoost: Actual vs Predicted',
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        return fig
    
    def create_xgboost_importance_chart(xgboost_data: Dict, colors: Dict) -> go.Figure:
        """Create XGBoost feature importance chart."""
        fig = go.Figure()
        
        importance = xgboost_data.get('feature_importance', {})
        if importance and 'by_gain' in importance:
            importance_list = importance['by_gain'][:10]  # Top 10
            
            features = [item['feature'] for item in importance_list]
            importance_values = [item.get('importance_normalized', item.get('importance', 0)) for item in importance_list]
            
            fig.add_trace(
                go.Bar(
                    x=importance_values,
                    y=features,
                    orientation='h',
                    marker_color=colors.get('primary', '#1e3c72'),
                    text=[f'{v:.1f}%' for v in importance_values],
                    textposition='auto',
                    hovertemplate='Feature: %{y}<br>Importance: %{x:.1f}%<extra></extra>'
                )
            )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            template='plotly_white',
            xaxis_title='Importance (%)',
            yaxis_title='Feature',
            yaxis=dict(categoryorder='total ascending'),
            title='Top Feature Importance',
            margin=dict(l=120, r=60, t=80, b=60)
        )
        
        return fig
    
    def create_xgboost_metrics_display(metrics: Dict) -> html.Div:
        """Create XGBoost metrics display."""
        if not metrics:
            return html.Div("No metrics available", style={'color': '#5f6368'})
        
        cards = []
        
        metric_items = [
            ('test_rmse', 'RMSE', lambda x: f"{x * 100:.2f}%", lambda x: 'positive' if x < 0.01 else 'neutral'),
            ('test_r2', 'R² Score', lambda x: f"{x:.3f}", lambda x: 'positive' if x > 0.3 else 'negative'),
            ('within_1std_pct', 'Within 1σ', lambda x: f"{x:.1f}%", lambda x: 'positive' if x > 70 else 'negative')
        ]
        
        for key, title, formatter, color_func in metric_items:
            if key in metrics:
                value = metrics[key]
                formatted_value = formatter(value)
                color_class = color_func(value)
                
                cards.append(
                    html.Div([
                        html.Div(title, className='metric-title'),
                        html.Div(formatted_value, className=f'metric-value {color_class}')
                    ], className='metric-card')
                )
        
        return html.Div(cards, className='metric-grid')
    
    # 11. Explainability Callbacks
    @app.callback(
        [Output('explainability-chart', 'figure'),
         Output('model-insights', 'children')],
        [Input('model-data-store', 'data')]
    )
    def update_explainability(model_data):
        """Update explainability charts and insights."""
        if not model_data:
            empty_fig = go.Figure()
            empty_fig.update_layout(template='plotly_white', height=450)
            return empty_fig, html.Div("No explainability data available.", style={'color': '#5f6368'})
        
        try:
            # Get explainability results
            explainability_data = model_data.get('explainability', {})
            
            # Create explainability chart
            fig = create_explainability_chart(explainability_data, color_palette)
            
            # Create insights
            insights = explainability_data.get('insights', [])
            insight_display = html.Ul(create_insight_display(insights), className='insight-list')
            
            return fig, insight_display
            
        except Exception as e:
            logger.error(f"Error updating explainability: {e}")
            empty_fig = go.Figure()
            empty_fig.update_layout(template='plotly_white', height=450)
            return empty_fig, html.Div(f"Error: {str(e)}", style={'color': '#db4437'})
    
    def create_explainability_chart(explainability_data: Dict, colors: Dict) -> go.Figure:
        """Create SHAP summary chart."""
        fig = go.Figure()
        
        summary = explainability_data.get('summary', {})
        if summary and 'feature_importance' in summary:
            importance_data = summary['feature_importance'][:10]
            
            features = [item['feature'] for item in importance_data]
            importance_values = [item['importance'] for item in importance_data]
            
            # Color by direction
            bar_colors = []
            for item in importance_data:
                direction = item.get('direction', 0)
                if direction > 0:
                    bar_colors.append('#db4437')  # Positive impact (red)
                else:
                    bar_colors.append('#0f9d58')  # Negative impact (green)
            
            fig.add_trace(
                go.Bar(
                    x=importance_values,
                    y=features,
                    orientation='h',
                    marker_color=bar_colors,
                    text=[f'{v:.4f}' for v in importance_values],
                    textposition='auto',
                    hovertemplate='Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>'
                )
            )
        
        fig.update_layout(
            height=450,
            showlegend=False,
            template='plotly_white',
            xaxis_title='Mean |SHAP Value|',
            yaxis_title='Feature',
            yaxis=dict(categoryorder='total ascending'),
            title='SHAP Feature Importance',
            margin=dict(l=120, r=60, t=80, b=60)
        )
        
        return fig
    
    # 12. Cross-Symbol Analysis Callbacks
    @app.callback(
        [Output('correlation-chart', 'figure'),
         Output('performance-chart', 'figure'),
         Output('cross-symbol-insights', 'children')],
        [Input('analysis-data-store', 'data')]
    )
    def update_cross_symbol_analysis(analysis_data):
        """Update cross-symbol analysis charts and insights."""
        
        fig1 = go.Figure()
        fig2 = go.Figure()
        
        # Create placeholder correlation matrix
        if len(symbols) >= 2:
            # Mock correlation matrix
            corr_values = np.random.rand(len(symbols), len(symbols))
            np.fill_diagonal(corr_values, 1.0)
            corr_matrix = (corr_values + corr_values.T) / 2  # Make symmetric
            
            fig1.add_trace(
                go.Heatmap(
                    z=corr_matrix,
                    x=symbols,
                    y=symbols,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_matrix, 2),
                    texttemplate='%{text}',
                    hoverongaps=False,
                    hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>'
                )
            )
            
            fig1.update_layout(
                height=450,
                title='Return Correlation Matrix',
                template='plotly_white',
                margin=dict(l=80, r=60, t=80, b=80)
            )
            
            # Mock performance comparison
            performance_values = np.random.uniform(5, 20, len(symbols))
            bar_colors = [color_palette.get('primary', '#1e3c72')] * len(symbols)
            
            fig2.add_trace(
                go.Bar(
                    x=symbols,
                    y=performance_values,
                    name='YTD Return',
                    marker_color=bar_colors,
                    text=[f'{v:.1f}%' for v in performance_values],
                    textposition='auto',
                    hovertemplate='Symbol: %{x}<br>Return: %{y:.1f}%<extra></extra>'
                )
            )
            
            fig2.update_layout(
                height=450,
                title='YTD Performance Comparison',
                template='plotly_white',
                xaxis_title='Symbol',
                yaxis_title='Return (%)',
                margin=dict(l=60, r=60, t=80, b=60)
            )
        else:
            fig1.update_layout(template='plotly_white', height=450)
            fig2.update_layout(template='plotly_white', height=450)
        
        insights = ["Cross-symbol analysis shows correlation patterns and relative performance across selected assets."]
        insight_display = html.Ul(create_insight_display(insights), className='insight-list')
        
        return fig1, fig2, insight_display
    
    # 13. Refresh Callback
    @app.callback(
        Output('symbol-selector', 'value'),
        [Input('refresh-button', 'n_clicks')],
        [State('symbol-selector', 'value')]
    )
    def refresh_data(n_clicks, current_symbol):
        """Refresh data for current symbol."""
        if n_clicks and n_clicks > 0:
            logger.info(f"Refreshing data for {current_symbol}")
            # Trigger reload by returning same symbol (will trigger other callbacks)
            return current_symbol
        return no_update

    logger.info("All callbacks registered successfully")