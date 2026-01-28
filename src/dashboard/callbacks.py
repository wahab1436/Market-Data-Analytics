"""
Dashboard Callbacks Module
Interactive callbacks for the Market Insight Platform dashboard
Read-only: All data is precomputed, no computation in callbacks
"""

from dash import Input, Output, State, callback_context, no_update
import pandas as pd
import numpy as np
from pathlib import Path
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


def register_callbacks(app, config: Dict[str, Any]):
    """Register all dashboard callbacks."""
    
    symbols = config['data']['symbols']
    color_palette = config['dashboard']['color_palette']
    data_paths = config['paths']
    
    # Helper function to load precomputed data
    def load_precomputed_data(symbol: str, data_type: str):
        """Load precomputed data from disk."""
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
            app.logger.error(f"Error loading {data_type} data for {symbol}: {e}")
            return None
    
    # Helper function to create KPI cards
    def create_kpi_display(metrics: Dict[str, Any]) -> List:
        """Create KPI display from metrics."""
        displays = []
        
        # Current Price
        current_price = metrics.get('current_price', 0)
        displays.append({
            'id': 'price-kpi',
            'value': f"${current_price:.2f}",
            'trend': metrics.get('price_change_1d', 0)
        })
        
        # Daily Return
        daily_return = metrics.get('price_change_1d', 0)
        displays.append({
            'id': 'return-kpi',
            'value': f"{daily_return:+.2f}%",
            'trend': "▲" if daily_return > 0 else "▼"
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
        avg_volume = metrics.get('avg_volume_20d', 0)
        volume_ratio = (volume / avg_volume) if avg_volume > 0 else 0
        
        if volume_ratio > 1:
            trend = f"{volume_ratio:.1f}x avg"
        else:
            trend = f"{1/volume_ratio:.1f}x below avg"
        
        displays.append({
            'id': 'volume-kpi',
            'value': f"{volume:,.0f}",
            'trend': trend
        })
        
        return displays
    
    # Helper function to create insight display
    def create_insight_display(insights: List[str]) -> List[html.Li]:
        """Create insight list items."""
        if not insights:
            return [html.Li("No insights available.", className="text-muted")]
        
        items = []
        for i, insight in enumerate(insights):
            items.append(html.Li(insight, className="insight-item mb-2"))
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
    
    # 2. KPI Update Callback
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
        """Update KPI cards with latest metrics."""
        if not analysis_data:
            return ["$--", "", "--%", "", "--%", "", "--", ""] * 2
        
        try:
            # Extract metrics from analysis data
            metrics = analysis_data.get('price', {}).get('metrics', {})
            
            # Create KPI displays
            displays = create_kpi_display(metrics)
            
            # Return formatted values
            return [
                displays[0]['value'], f"1d: {displays[0]['trend']:+.2f}%" if displays[0]['trend'] != 0 else "",
                displays[1]['value'], displays[1]['trend'],
                displays[2]['value'], displays[2]['trend'],
                displays[3]['value'], displays[3]['trend']
            ]
        except Exception as e:
            app.logger.error(f"Error updating KPIs: {e}")
            return ["$--", "", "--%", "", "--%", "", "--", ""] * 2
    
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
            return go.Figure(), "No data available.", "No metrics available."
        
        try:
            # Convert featured data back to DataFrame
            featured_df = pd.DataFrame(featured_data)
            if 'date' in featured_df.columns:
                featured_df['date'] = pd.to_datetime(featured_df['date'])
                featured_df = featured_df.set_index('date')
            
            # Get price analysis results
            price_analysis = analysis_data.get('price', {})
            
            # Create price chart
            fig = create_price_chart(featured_df, price_analysis)
            
            # Create insights
            insights = price_analysis.get('insights', [])
            insight_display = create_insight_display(insights)
            
            # Create metrics display
            metrics = price_analysis.get('metrics', {})
            metrics_display = create_metrics_display(metrics)
            
            return fig, insight_display, metrics_display
            
        except Exception as e:
            app.logger.error(f"Error updating price analysis: {e}")
            return go.Figure(), f"Error: {str(e)}", "Error loading metrics"
    
    def create_price_chart(df: pd.DataFrame, price_analysis: Dict) -> go.Figure:
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
                line=dict(color=color_palette['primary'], width=2),
                hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add moving averages if available
        for ma in [20, 50, 200]:
            ma_col = f'ma_{ma}d'
            if ma_col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[ma_col],
                        mode='lines',
                        name=f'{ma}-day MA',
                        line=dict(color=color_palette['secondary'], width=1.5, dash='dash'),
                        hovertemplate=f'{ma}-day MA: $%{{y:.2f}}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Daily returns
        if 'return' in df.columns:
            returns = df['return'] * 100
            colors = np.where(returns >= 0, color_palette['success'], color_palette['danger'])
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=returns,
                    name='Daily Return %',
                    marker_color=colors,
                    hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)
        
        return fig
    
    def create_metrics_display(metrics: Dict[str, float]) -> List:
        """Create metrics display cards."""
        if not metrics:
            return ["No metrics available"]
        
        metric_cards = []
        
        # Price metrics
        if 'current_price' in metrics:
            metric_cards.append(
                html.Div([
                    html.H6("Current Price", className="metric-title"),
                    html.H4(f"${metrics['current_price']:.2f}", className="metric-value")
                ], className="metric-card")
            )
        
        if 'price_change_30d' in metrics:
            change = metrics['price_change_30d']
            metric_cards.append(
                html.Div([
                    html.H6("30-day Change", className="metric-title"),
                    html.H4(f"{change:+.1f}%", className="metric-value"),
                    html.Span(f"{'▲' if change > 0 else '▼'}", 
                            className=f"metric-change {'positive' if change > 0 else 'negative'}")
                ], className="metric-card")
            )
        
        # Volatility metrics
        if 'annualized_volatility' in metrics:
            vol = metrics['annualized_volatility']
            metric_cards.append(
                html.Div([
                    html.H6("Annualized Vol", className="metric-title"),
                    html.H4(f"{vol:.1f}%", className="metric-value"),
                    html.Span("High" if vol > 30 else "Low" if vol < 15 else "Moderate",
                            className="metric-change")
                ], className="metric-card")
            )
        
        # Risk metrics
        if 'max_drawdown' in metrics:
            dd = metrics['max_drawdown']
            metric_cards.append(
                html.Div([
                    html.H6("Max Drawdown", className="metric-title"),
                    html.H4(f"{dd:.1f}%", className="metric-value"),
                    html.Span("Severe" if dd < -20 else "Moderate",
                            className="metric-change")
                ], className="metric-card")
            )
        
        return metric_cards
    
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
            return go.Figure(), go.Figure(), "No data available."
        
        try:
            # Convert featured data back to DataFrame
            featured_df = pd.DataFrame(featured_data)
            if 'date' in featured_df.columns:
                featured_df['date'] = pd.to_datetime(featured_df['date'])
                featured_df = featured_df.set_index('date')
            
            # Get volatility analysis results
            volatility_analysis = analysis_data.get('volatility', {})
            
            # Create volatility charts
            timeline_fig = create_volatility_timeline(featured_df)
            distribution_fig = create_volatility_distribution(featured_df)
            
            # Create insights
            insights = volatility_analysis.get('insights', [])
            insight_display = create_insight_display(insights)
            
            return timeline_fig, distribution_fig, insight_display
            
        except Exception as e:
            app.logger.error(f"Error updating volatility analysis: {e}")
            return go.Figure(), go.Figure(), f"Error: {str(e)}"
    
    def create_volatility_timeline(df: pd.DataFrame) -> go.Figure:
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
                    line=dict(color=color_palette['primary'], width=2),
                    fill='tozeroy',
                    fillcolor='rgba(46, 134, 171, 0.1)',
                    hovertemplate='Date: %{x}<br>Volatility: %{y:.1f}%<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add mean line
            vol_mean = rolling_vol.mean()
            fig.add_hline(
                y=vol_mean,
                line_width=1,
                line_dash="dash",
                line_color=color_palette['text'],
                annotation_text=f"Mean: {vol_mean:.1f}%",
                row=1, col=1
            )
            
            # Daily returns
            returns = df['return'] * 100
            colors = np.where(returns >= 0, color_palette['success'], color_palette['danger'])
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=returns,
                    name='Daily Return',
                    marker_color=colors,
                    hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)
        
        return fig
    
    def create_volatility_distribution(df: pd.DataFrame) -> go.Figure:
        """Create volatility distribution chart."""
        fig = go.Figure()
        
        if 'return' in df.columns:
            returns = df['return'] * 100
            
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    nbinsx=50,
                    name='Return Distribution',
                    marker_color=color_palette['primary'],
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
                line_color=color_palette['text'],
                annotation_text=f"Mean: {mean_return:.2f}%"
            )
            
            fig.add_vline(
                x=mean_return + std_return,
                line_width=1,
                line_dash="dot",
                line_color=color_palette['warning']
            )
            
            fig.add_vline(
                x=mean_return - std_return,
                line_width=1,
                line_dash="dot",
                line_color=color_palette['warning']
            )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            template='plotly_white',
            xaxis_title='Daily Return (%)',
            yaxis_title='Frequency'
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
            return go.Figure(), "No data available.", "No relationship data."
        
        try:
            # Convert featured data back to DataFrame
            featured_df = pd.DataFrame(featured_data)
            if 'date' in featured_df.columns:
                featured_df['date'] = pd.to_datetime(featured_df['date'])
                featured_df = featured_df.set_index('date')
            
            # Get volume analysis results
            volume_analysis = analysis_data.get('volume', {})
            
            # Create volume chart
            fig = create_volume_chart(featured_df)
            
            # Create insights
            insights = volume_analysis.get('insights', [])
            insight_display = create_insight_display(insights)
            
            # Create relationship metrics
            relationship_display = create_volume_relationship_display(volume_analysis)
            
            return fig, insight_display, relationship_display
            
        except Exception as e:
            app.logger.error(f"Error updating volume analysis: {e}")
            return go.Figure(), f"Error: {str(e)}", "Error loading relationship data"
    
    def create_volume_chart(df: pd.DataFrame) -> go.Figure:
        """Create volume-price chart."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=("Price & Volume", "Volume Ratio (vs 20-day MA)")
        )
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['close'],
                mode='lines',
                name='Close Price',
                line=dict(color=color_palette['primary'], width=2),
                yaxis='y1',
                hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Volume bars (secondary y-axis)
        if 'volume' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    name='Volume',
                    marker_color='rgba(162, 59, 114, 0.6)',
                    yaxis='y2',
                    hovertemplate='Volume: %{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Volume ratio
        if 'volume_ratio_20d' in df.columns:
            volume_ratio = df['volume_ratio_20d']
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=volume_ratio,
                    mode='lines',
                    name='Volume Ratio (vs 20-day MA)',
                    line=dict(color=color_palette['success'], width=1.5),
                    hovertemplate='Volume Ratio: %{y:.2f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add horizontal line at 1.0
            fig.add_hline(
                y=1.0,
                line_width=1,
                line_dash="dash",
                line_color=color_palette['text'],
                row=2, col=1
            )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white',
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="right",
                showgrid=False
            )
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume Ratio", row=2, col=1)
        
        return fig
    
    def create_volume_relationship_display(volume_analysis: Dict) -> List:
        """Create volume-price relationship display."""
        if not volume_analysis:
            return ["No relationship data available"]
        
        metrics = volume_analysis.get('metrics', {})
        
        cards = []
        
        # Volume-return correlation
        if 'current_volume_return_corr' in metrics:
            corr = metrics['current_volume_return_corr']
            cards.append(
                html.Div([
                    html.H6("Volume-Return Correlation", className="metric-title"),
                    html.H4(f"{corr:.3f}", className="metric-value"),
                    html.Span("Strong" if abs(corr) > 0.3 else "Weak",
                            className="metric-change")
                ], className="metric-card")
            )
        
        # Volume ratio
        if 'volume_ratio_current_vs_avg' in metrics:
            ratio = metrics['volume_ratio_current_vs_avg']
            cards.append(
                html.Div([
                    html.H6("Current vs Avg Volume", className="metric-title"),
                    html.H4(f"{ratio:.1f}x", className="metric-value"),
                    html.Span("High" if ratio > 1.5 else "Low" if ratio < 0.7 else "Normal",
                            className="metric-change")
                ], className="metric-card")
            )
        
        # Extreme volume days
        if 'pct_extreme_volume_days' in metrics:
            pct = metrics['pct_extreme_volume_days']
            cards.append(
                html.Div([
                    html.H6("Extreme Volume Days", className="metric-title"),
                    html.H4(f"{pct:.1f}%", className="metric-value"),
                    html.Span("Frequent" if pct > 10 else "Rare",
                            className="metric-change")
                ], className="metric-card")
            )
        
        return cards
    
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
            return go.Figure(), "No data available."
        
        try:
            # Convert featured data back to DataFrame
            featured_df = pd.DataFrame(featured_data)
            if 'date' in featured_df.columns:
                featured_df['date'] = pd.to_datetime(featured_df['date'])
                featured_df = featured_df.set_index('date')
            
            # Get similarity analysis results
            similarity_analysis = analysis_data.get('similarity', {})
            
            # Create similarity chart
            fig = create_similarity_chart(featured_df, similarity_analysis)
            
            # Create insights
            insights = similarity_analysis.get('insights', [])
            insight_display = create_insight_display(insights)
            
            return fig, insight_display
            
        except Exception as e:
            app.logger.error(f"Error updating similarity analysis: {e}")
            return go.Figure(), f"Error: {str(e)}"
    
    def create_similarity_chart(df: pd.DataFrame, similarity_analysis: Dict) -> go.Figure:
        """Create similarity pattern chart."""
        fig = go.Figure()
        
        # Plot recent price pattern
        lookback = 20
        if len(df) >= lookback:
            recent_prices = df['close'].iloc[-lookback:]
            
            # Normalize for pattern comparison
            min_price = recent_prices.min()
            max_price = recent_prices.max()
            normalized_prices = (recent_prices - min_price) / (max_price - min_price)
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(lookback)),
                    y=normalized_prices,
                    mode='lines+markers',
                    name='Current Pattern',
                    line=dict(color=color_palette['primary'], width=3),
                    marker=dict(size=8),
                    hovertemplate='Day: %{x}<br>Normalized: %{y:.3f}<extra></extra>'
                )
            )
            
            # Add analog patterns if available
            analogs = similarity_analysis.get('top_analogs', [])
            for i, analog in enumerate(analogs[:3]):  # Top 3 analogs
                # In a real implementation, you would plot the actual analog patterns
                # This is a simplified placeholder
                fig.add_trace(
                    go.Scatter(
                        x=list(range(lookback)),
                        y=[0.8 - i*0.1] * lookback,  # Placeholder
                        mode='lines',
                        name=f'Analog {i+1}',
                        line=dict(color=color_palette['secondary'], width=2, dash='dash'),
                        hovertemplate=f'Analog {i+1}<extra></extra>'
                    )
                )
        
        fig.update_layout(
            height=500,
            showlegend=True,
            template='plotly_white',
            xaxis_title='Trading Days',
            yaxis_title='Normalized Price',
            title='Current Pattern vs Historical Analogs'
        )
        
        return fig
    
    # 7. Model Tabs Callback
    @app.callback(
        Output('model-tab-content', 'children'),
        [Input('model-tabs', 'active_tab')]
    )
    def update_model_tab(active_tab):
        """Update model tab content."""
        from .layout import create_model_tab_content
        return create_model_tab_content(active_tab)
    
    # 8. Model Analysis Callbacks
    @app.callback(
        [Output('regression-chart', 'figure'),
         Output('regression-metrics-chart', 'figure'),
         Output('regression-insights', 'children')],
        [Input('model-data-store', 'data')]
    )
    def update_regression_analysis(model_data):
        """Update regression analysis charts and insights."""
        if not model_data:
            return go.Figure(), go.Figure(), "No model data available."
        
        try:
            # Get regression results
            regression_data = model_data.get('regression', {})
            
            # Create regression charts
            pred_fig = create_regression_prediction_chart(regression_data)
            metrics_fig = create_regression_metrics_chart(regression_data)
            
            # Create insights
            insights = regression_data.get('insights', [])
            insight_display = create_insight_display(insights)
            
            return pred_fig, metrics_fig, insight_display
            
        except Exception as e:
            app.logger.error(f"Error updating regression analysis: {e}")
            return go.Figure(), go.Figure(), f"Error: {str(e)}"
    
    def create_regression_prediction_chart(regression_data: Dict) -> go.Figure:
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
                        line=dict(color=color_palette['primary'], width=2),
                        hovertemplate='Date: %{x}<br>Actual: %{y:.2f}%<extra></extra>'
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=pred_df['date'],
                        y=pred_df['predicted'] * 100,
                        mode='lines',
                        name='Predicted',
                        line=dict(color=color_palette['success'], width=2, dash='dash'),
                        hovertemplate='Date: %{x}<br>Predicted: %{y:.2f}%<extra></extra>'
                    )
                )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            template='plotly_white',
            xaxis_title='Date',
            yaxis_title='Absolute Return (%)',
            title='Regression: Actual vs Predicted'
        )
        
        return fig
    
    def create_regression_metrics_chart(regression_data: Dict) -> go.Figure:
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
                rmse_values.append(metrics[rmse_key] * 100)  # Convert to percentage
                r2_values.append(metrics[r2_key])
        
        if model_names:
            # RMSE bars
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=rmse_values,
                    name='RMSE (%)',
                    marker_color=color_palette['primary'],
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
                    line=dict(color=color_palette['success'], width=3),
                    marker=dict(size=10),
                    text=[f'{v:.3f}' for v in r2_values],
                    hovertemplate='R²: %{y:.3f}<extra></extra>'
                )
            )
            
            fig.update_layout(
                yaxis=dict(
                    title='RMSE (%)',
                    titlefont=dict(color=color_palette['primary'])
                ),
                yaxis2=dict(
                    title='R² Score',
                    titlefont=dict(color=color_palette['success']),
                    overlaying='y',
                    side='right'
                )
            )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            template='plotly_white',
            xaxis_title='Model',
            title='Model Performance Comparison'
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
            return go.Figure(), "No analog data.", "No insights available."
        
        try:
            # Get KNN results
            knn_data = model_data.get('knn', {})
            
            # Create KNN chart
            fig = create_knn_chart(knn_data)
            
            # Create analog display
            analogs = knn_data.get('analogs', {}).get('top_analogs', [])
            analog_display = create_analog_display(analogs)
            
            # Create insights
            insights = knn_data.get('insights', [])
            insight_display = create_insight_display(insights)
            
            return fig, analog_display, insight_display
            
        except Exception as e:
            app.logger.error(f"Error updating KNN analysis: {e}")
            return go.Figure(), f"Error: {str(e)}", "Error loading insights"
    
    def create_knn_chart(knn_data: Dict) -> go.Figure:
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
                        line=dict(color=color_palette['primary'], width=2),
                        hovertemplate='Date: %{x}<br>Actual: %{y:.2f}%<extra></extra>'
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=pred_df['date'],
                        y=pred_df['predicted'] * 100,
                        mode='lines',
                        name='KNN Predicted',
                        line=dict(color=color_palette['secondary'], width=2, dash='dash'),
                        hovertemplate='Date: %{x}<br>Predicted: %{y:.2f}%<extra></extra>'
                    )
                )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            template='plotly_white',
            xaxis_title='Date',
            yaxis_title='Absolute Return (%)',
            title='KNN: Actual vs Predicted'
        )
        
        return fig
    
    def create_analog_display(analogs: List) -> List:
        """Create analog display."""
        if not analogs:
            return ["No historical analogs found"]
        
        display_items = []
        for i, analog in enumerate(analogs[:3]):  # Top 3 analogs
            display_items.append(
                html.Div([
                    html.H6(f"Analog {i+1}", className="mb-1"),
                    html.P(f"Date: {analog.get('date', 'N/A')}", className="small mb-1"),
                    html.P(f"Similarity: {analog.get('similarity', 0):.3f}", className="small mb-1"),
                    html.P(f"Days Apart: {analog.get('days_apart', 0)}", className="small")
                ], className="mb-3 p-2 border rounded")
            )
        
        return display_items
    
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
            return go.Figure(), go.Figure(), "No XGBoost data available."
        
        try:
            # Get XGBoost results
            xgboost_data = model_data.get('xgboost', {})
            
            # Create XGBoost charts
            pred_fig = create_xgboost_prediction_chart(xgboost_data)
            importance_fig = create_xgboost_importance_chart(xgboost_data)
            
            # Create metrics display
            metrics = xgboost_data.get('metrics', {})
            metrics_display = create_xgboost_metrics_display(metrics)
            
            return pred_fig, importance_fig, metrics_display
            
        except Exception as e:
            app.logger.error(f"Error updating XGBoost analysis: {e}")
            return go.Figure(), go.Figure(), f"Error: {str(e)}"
    
    def create_xgboost_prediction_chart(xgboost_data: Dict) -> go.Figure:
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
                        line=dict(color=color_palette['primary'], width=2),
                        hovertemplate='Date: %{x}<br>Actual: %{y:.2f}%<extra></extra>'
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=pred_df['date'],
                        y=pred_df['predicted'] * 100,
                        mode='lines',
                        name='XGBoost Predicted',
                        line=dict(color=color_palette['warning'], width=2, dash='dash'),
                        hovertemplate='Date: %{x}<br>Predicted: %{y:.2f}%<extra></extra>'
                    )
                )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            template='plotly_white',
            xaxis_title='Date',
            yaxis_title='Absolute Return (%)',
            title='XGBoost: Actual vs Predicted'
        )
        
        return fig
    
    def create_xgboost_importance_chart(xgboost_data: Dict) -> go.Figure:
        """Create XGBoost feature importance chart."""
        fig = go.Figure()
        
        importance = xgboost_data.get('feature_importance', {})
        if importance and 'by_gain' in importance:
            importance_list = importance['by_gain']
            
            features = [item['feature'] for item in importance_list[:10]]  # Top 10
            importance_values = [item['importance_normalized'] for item in importance_list[:10]]
            
            fig.add_trace(
                go.Bar(
                    x=importance_values,
                    y=features,
                    orientation='h',
                    marker_color=color_palette['primary'],
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
            title='Top Feature Importance'
        )
        
        return fig
    
    def create_xgboost_metrics_display(metrics: Dict) -> List:
        """Create XGBoost metrics display."""
        if not metrics:
            return ["No metrics available"]
        
        cards = []
        
        if 'test_rmse' in metrics:
            rmse = metrics['test_rmse'] * 100
            cards.append(
                html.Div([
                    html.H6("RMSE", className="metric-title"),
                    html.H4(f"{rmse:.2f}%", className="metric-value"),
                    html.Span("Good" if rmse < 1.0 else "Moderate",
                            className="metric-change")
                ], className="metric-card")
            )
        
        if 'test_r2' in metrics:
            r2 = metrics['test_r2']
            cards.append(
                html.Div([
                    html.H6("R² Score", className="metric-title"),
                    html.H4(f"{r2:.3f}", className="metric-value"),
                    html.Span("Strong" if r2 > 0.3 else "Weak",
                            className="metric-change")
                ], className="metric-card")
            )
        
        if 'within_1std_pct' in metrics:
            within = metrics['within_1std_pct']
            cards.append(
                html.Div([
                    html.H6("Within 1σ", className="metric-title"),
                    html.H4(f"{within:.1f}%", className="metric-value"),
                    html.Span("Accurate" if within > 70 else "Variable",
                            className="metric-change")
                ], className="metric-card")
            )
        
        return cards
    
    # 11. Explainability Callbacks
    @app.callback(
        [Output('explainability-chart', 'figure'),
         Output('model-insights', 'children')],
        [Input('model-data-store', 'data')]
    )
    def update_explainability(model_data):
        """Update explainability charts and insights."""
        if not model_data:
            return go.Figure(), "No explainability data available."
        
        try:
            # Get explainability results
            explainability_data = model_data.get('explainability', {})
            
            # Create explainability chart
            fig = create_explainability_chart(explainability_data)
            
            # Create insights
            insights = explainability_data.get('insights', [])
            insight_display = create_insight_display(insights)
            
            return fig, insight_display
            
        except Exception as e:
            app.logger.error(f"Error updating explainability: {e}")
            return go.Figure(), f"Error: {str(e)}"
    
    def create_explainability_chart(explainability_data: Dict) -> go.Figure:
        """Create SHAP summary chart."""
        fig = go.Figure()
        
        summary = explainability_data.get('summary', {})
        if summary and 'feature_importance' in summary:
            importance_data = summary['feature_importance']
            
            features = [item['feature'] for item in importance_data[:10]]
            importance_values = [item['importance'] for item in importance_data[:10]]
            
            # Color by direction
            colors = []
            for item in importance_data[:10]:
                if item['direction'] > 0:
                    colors.append(color_palette['danger'])  # Positive impact
                else:
                    colors.append(color_palette['success'])  # Negative impact
            
            fig.add_trace(
                go.Bar(
                    x=importance_values,
                    y=features,
                    orientation='h',
                    marker_color=colors,
                    text=[f'{v:.4f}' for v in importance_values],
                    textposition='auto',
                    hovertemplate='Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>'
                )
            )
        
        fig.update_layout(
            height=500,
            showlegend=False,
            template='plotly_white',
            xaxis_title='Mean |SHAP Value|',
            yaxis_title='Feature',
            yaxis=dict(categoryorder='total ascending'),
            title='SHAP Feature Importance'
        )
        
        return fig
    
    # 12. Cross-Symbol Analysis Callbacks
    @app.callback(
        [Output('correlation-chart', 'figure'),
         Output('performance-chart', 'figure'),
         Output('cross-symbol-insights', 'children')],
        [Input('analysis-data-store', 'data'),
         Input('model-data-store', 'data')]
    )
    def update_cross_symbol_analysis(analysis_data, model_data):
        """Update cross-symbol analysis charts and insights."""
        # This is a simplified implementation
        # In production, you would load cross-symbol data
        
        fig1 = go.Figure()
        fig2 = go.Figure()
        
        # Create placeholder correlation matrix
        symbols = config['data']['symbols']
        if len(symbols) >= 2:
            # Mock correlation matrix
            corr_matrix = [[1.0, 0.7, 0.5],
                          [0.7, 1.0, 0.6],
                          [0.5, 0.6, 1.0]][:len(symbols)][:len(symbols)]
            
            fig1.add_trace(
                go.Heatmap(
                    z=corr_matrix,
                    x=symbols,
                    y=symbols,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_matrix, 2),
                    texttemplate='%{text}',
                    hoverongaps=False
                )
            )
            
            fig1.update_layout(
                height=400,
                title='Return Correlation Matrix'
            )
            
            # Mock performance comparison
            fig2.add_trace(
                go.Bar(
                    x=symbols,
                    y=[12.5, 8.3, 15.2][:len(symbols)],  # Mock returns
                    name='YTD Return',
                    marker_color=color_palette['primary']
                )
            )
            
            fig2.update_layout(
                height=400,
                title='YTD Performance Comparison'
            )
        
        insights = ["Cross-symbol analysis shows correlation patterns and relative performance."]
        insight_display = create_insight_display(insights)
        
        return fig1, fig2, insight_display
    
    # 13. Refresh Callback
    @app.callback(
        [Output('symbol-selector', 'value'),
         Output('analysis-data-store', 'clear_data'),
         Output('model-data-store', 'clear_data'),
         Output('featured-data-store', 'clear_data')],
        [Input('refresh-button', 'n_clicks')],
        [State('symbol-selector', 'value')]
    )
    def refresh_data(n_clicks, current_symbol):
        """Refresh data for current symbol."""
        if n_clicks > 0:
            # Trigger data reload by clearing stores
            return current_symbol, True, True, True
        return current_symbol, False, False, False
    
    # 14. Navigation Callback (smooth scroll)
    @app.callback(
        Output('url', 'pathname'),
        [Input('price-section-link', 'n_clicks'),
         Input('volatility-section-link', 'n_clicks'),
         Input('volume-section-link', 'n_clicks'),
         Input('similarity-section-link', 'n_clicks'),
         Input('models-section-link', 'n_clicks'),
         Input('explainability-section-link', 'n_clicks')]
    )
    def navigate_to_section(price_clicks, vol_clicks, volume_clicks, 
                           similarity_clicks, models_clicks, explain_clicks):
        """Handle navigation between sections."""
        ctx = callback_context
        
        if not ctx.triggered:
            return no_update
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        section_map = {
            'price-section-link': '#price-section',
            'volatility-section-link': '#volatility-section',
            'volume-section-link': '#volume-section',
            'similarity-section-link': '#similarity-section',
            'models-section-link': '#models-section',
            'explainability-section-link': '#explainability-section'
        }
        
        return section_map.get(button_id, '#price-section')
