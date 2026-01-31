"""
Price Analysis Module
Analyzes price trends, levels, and patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib


class PriceAnalysis:
    """Analyzes price behavior and trends."""
    
    def __init__(self, config: Dict[str, Any], logger):
        """Initialize price analyzer."""
        self.config = config
        self.logger = logger
        self.artifacts_path = Path(config['paths']['artifacts'])
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive price analysis."""
        results = {}
        
        by_symbol = data['by_symbol']
        
        for symbol, df in by_symbol.items():
            self.logger.info(f"Analyzing price for {symbol}")
            
            symbol_results = {
                'charts': {},
                'insights': [],
                'metrics': {}
            }
            
            # Create price charts
            price_charts = self._create_price_charts(df, symbol)
            symbol_results['charts'].update(price_charts)
            
            # Calculate metrics
            metrics = self._calculate_price_metrics(df)
            symbol_results['metrics'].update(metrics)
            
            # Generate insights
            insights = self._generate_price_insights(df, metrics)
            symbol_results['insights'].extend(insights)
            
            # Trend analysis
            trend_results = self._analyze_trends(df)
            symbol_results.update(trend_results)
            
            results[symbol] = symbol_results
            
            # Save individual symbol results
            self._save_symbol_results(symbol, symbol_results)
        
        # Cross-symbol analysis
        if len(by_symbol) > 1:
            cross_results = self._analyze_cross_symbol_price(by_symbol)
            results['cross_symbol'] = cross_results
        
        return results
    
    def _save_symbol_results(self, symbol: str, results: Dict[str, Any]):
        """Save analysis results for a symbol to artifacts folder."""
        try:
            # Create a serializable copy (remove figures which can't be pickled easily)
            serializable_results = {
                'metrics': results.get('metrics', {}),
                'insights': results.get('insights', []),
                'golden_cross_count': results.get('golden_cross_count', 0),
                'death_cross_count': results.get('death_cross_count', 0),
                'current_trend': results.get('current_trend', 'Unknown'),
                'support_level': results.get('support_level', 0),
                'resistance_level': results.get('resistance_level', 0),
                'current_vs_support': results.get('current_vs_support', 0),
                'current_vs_resistance': results.get('current_vs_resistance', 0)
            }
            
            # Save to pickle file
            output_file = self.artifacts_path / f"{symbol}_analysis_results.pkl"
            joblib.dump({'price': serializable_results}, output_file)
            
            self.logger.info(f"Saved price analysis results for {symbol} to {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save price analysis results for {symbol}: {e}")
    
    def _create_price_charts(self, df: pd.DataFrame, symbol: str) -> Dict[str, go.Figure]:
        """Create price visualization charts."""
        charts = {}
        
        # 1. Price with Moving Averages
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{symbol} Price with Moving Averages', 'Daily Returns')
        )
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['close'],
                mode='lines',
                name='Close Price',
                line=dict(color=self.config['dashboard']['color_palette']['primary'], width=2),
                hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Moving averages
        ma_colors = {
            20: self.config['dashboard']['color_palette']['secondary'],
            50: self.config['dashboard']['color_palette']['success'],
            200: self.config['dashboard']['color_palette']['warning']
        }
        
        for ma in [20, 50, 200]:
            if f'ma_{ma}d' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[f'ma_{ma}d'],
                        mode='lines',
                        name=f'{ma}-day MA',
                        line=dict(color=ma_colors[ma], width=1.5, dash='dash'),
                        hovertemplate=f'{ma}-day MA: $%{{y:.2f}}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Daily returns
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['return'] * 100,
                name='Daily Return %',
                marker_color=np.where(df['return'] >= 0, 
                                    self.config['dashboard']['color_palette']['success'],
                                    self.config['dashboard']['color_palette']['danger']),
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
        
        charts['price_with_mas'] = fig
        
        # 2. Price Distribution
        fig2 = go.Figure()
        
        # Histogram of returns
        fig2.add_trace(
            go.Histogram(
                x=df['return'] * 100,
                nbinsx=50,
                name='Return Distribution',
                marker_color=self.config['dashboard']['color_palette']['primary'],
                opacity=0.7,
                hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
            )
        )
        
        # Add vertical line at 0
        fig2.add_vline(
            x=0,
            line_width=2,
            line_dash="dash",
            line_color=self.config['dashboard']['color_palette']['text']
        )
        
        fig2.update_layout(
            title=f'{symbol} Return Distribution',
            xaxis_title='Daily Return (%)',
            yaxis_title='Frequency',
            height=400,
            showlegend=False,
            template='plotly_white'
        )
        
        charts['return_distribution'] = fig2
        
        return charts
    
    def _calculate_price_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate key price metrics."""
        metrics = {}
        
        if len(df) < 2:
            return metrics
        
        # Basic metrics
        metrics['current_price'] = float(df['close'].iloc[-1])
        
        # Handle edge cases for price changes
        if len(df) >= 1:
            metrics['price_change_1d'] = float(df['return'].iloc[-1] * 100) if 'return' in df.columns else 0.0
        
        if len(df) >= 5:
            metrics['price_change_5d'] = float(((df['close'].iloc[-1] / df['close'].iloc[-5]) - 1) * 100)
        else:
            metrics['price_change_5d'] = 0.0
            
        if len(df) >= 30:
            metrics['price_change_30d'] = float(((df['close'].iloc[-1] / df['close'].iloc[-30]) - 1) * 100)
        else:
            metrics['price_change_30d'] = 0.0
        
        # Volatility metrics
        metrics['annualized_volatility'] = float(df['return'].std() * np.sqrt(252) * 100)
        
        if df['return'].std() > 0:
            metrics['sharpe_ratio'] = float(df['return'].mean() / df['return'].std() * np.sqrt(252))
        else:
            metrics['sharpe_ratio'] = 0.0
        
        # Risk metrics
        metrics['max_drawdown'] = self._calculate_max_drawdown(df['close'])
        metrics['value_at_risk_95'] = float(np.percentile(df['return'], 5) * 100)
        
        # Distribution metrics
        metrics['skewness'] = float(df['return'].skew())
        metrics['kurtosis'] = float(df['return'].kurtosis())
        
        # Price level metrics
        if 'rolling_high_20d' in df.columns and 'rolling_low_20d' in df.columns:
            if df['rolling_high_20d'].iloc[-1] > 0:
                metrics['vs_20d_high'] = float(((df['close'].iloc[-1] / df['rolling_high_20d'].iloc[-1]) - 1) * 100)
            if df['rolling_low_20d'].iloc[-1] > 0:
                metrics['vs_20d_low'] = float(((df['close'].iloc[-1] / df['rolling_low_20d'].iloc[-1]) - 1) * 100)
        
        # Volume metrics
        if 'volume' in df.columns:
            metrics['current_volume'] = float(df['volume'].iloc[-1])
            if len(df) >= 20:
                metrics['avg_volume_20d'] = float(df['volume'].tail(20).mean())
        
        return metrics
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return float(drawdown.min() * 100)  # %
    
    def _generate_price_insights(self, df: pd.DataFrame, metrics: Dict[str, float]) -> List[str]:
        """Generate textual insights from price analysis."""
        insights = []
        
        if not metrics:
            return insights
        
        # Trend insight
        price_change_30d = metrics.get('price_change_30d', 0)
        if price_change_30d > 10:
            insights.append(f"Strong uptrend: Price up {price_change_30d:.1f}% over last 30 days.")
        elif price_change_30d < -10:
            insights.append(f"Strong downtrend: Price down {abs(price_change_30d):.1f}% over last 30 days.")
        else:
            insights.append(f"Sideways movement: Price change of {price_change_30d:.1f}% over last 30 days.")
        
        # Volatility insight
        volatility = metrics.get('annualized_volatility', 0)
        if volatility > 40:
            insights.append(f"High volatility environment: {volatility:.1f}% annualized volatility.")
        elif volatility < 20:
            insights.append(f"Low volatility environment: {volatility:.1f}% annualized volatility.")
        
        # Risk insight
        max_dd = metrics.get('max_drawdown', 0)
        if max_dd < -20:
            insights.append(f"Significant drawdown risk: Maximum drawdown of {abs(max_dd):.1f}%.")
        
        # Distribution insight
        skew = metrics.get('skewness', 0)
        if skew > 0.5:
            insights.append("Return distribution shows positive skew (more large gains than losses).")
        elif skew < -0.5:
            insights.append("Return distribution shows negative skew (more large losses than gains).")
        
        # Price level insight
        vs_high = metrics.get('vs_20d_high', 0)
        if vs_high > -2:  # Within 2% of 20-day high
            insights.append("Price trading near 20-day highs, showing strong momentum.")
        
        return insights
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price trends and patterns."""
        results = {}
        
        if len(df) < 50:
            return results
        
        # Moving average cross analysis
        if 'ma_20d' in df.columns and 'ma_50d' in df.columns:
            ma_20 = df['ma_20d']
            ma_50 = df['ma_50d']
            
            # Golden cross / death cross
            golden_cross = ((ma_20 > ma_50) & (ma_20.shift(1) <= ma_50.shift(1))).sum()
            death_cross = ((ma_20 < ma_50) & (ma_20.shift(1) >= ma_50.shift(1))).sum()
            
            results['golden_cross_count'] = int(golden_cross)
            results['death_cross_count'] = int(death_cross)
            
            # Current trend
            if ma_20.iloc[-1] > ma_50.iloc[-1]:
                results['current_trend'] = 'Bullish (20-day MA > 50-day MA)'
            else:
                results['current_trend'] = 'Bearish (20-day MA < 50-day MA)'
        
        # Support/resistance levels
        if 'rolling_high_20d' in df.columns and 'rolling_low_20d' in df.columns:
            current_high = df['rolling_high_20d'].iloc[-1]
            current_low = df['rolling_low_20d'].iloc[-1]
            current_close = df['close'].iloc[-1]
            
            results['support_level'] = float(current_low)
            results['resistance_level'] = float(current_high)
            results['current_vs_support'] = float(((current_close / current_low) - 1) * 100) if current_low > 0 else 0.0
            results['current_vs_resistance'] = float(((current_close / current_high) - 1) * 100) if current_high > 0 else 0.0
        
        return results
    
    def _analyze_cross_symbol_price(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze price relationships between symbols."""
        results = {}
        
        symbols = list(data.keys())
        if len(symbols) < 2:
            return results
        
        # Correlation matrix
        returns_data = {}
        for symbol, df in data.items():
            returns_data[symbol] = df['return']
        
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()
        
        results['correlation_matrix'] = correlation_matrix.to_dict()
        
        # Create correlation heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverongaps=False,
            hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Return Correlation Matrix',
            height=400,
            width=500,
            template='plotly_white'
        )
        
        results['correlation_chart'] = fig
        
        # Generate insights
        insights = []
        
        # Find highest correlation pair
        if len(symbols) >= 2:
            corr_values = correlation_matrix.unstack()
            corr_values = corr_values[corr_values < 1]  # Remove self-correlations
            
            if len(corr_values) > 0:
                max_corr = corr_values.max()
                min_corr = corr_values.min()
                
                max_pair = corr_values[corr_values == max_corr].index[0]
                min_pair = corr_values[corr_values == min_corr].index[0]
                
                insights.append(f"Highest correlation: {max_pair[0]}-{max_pair[1]} ({max_corr:.2f})")
                insights.append(f"Lowest correlation: {min_pair[0]}-{min_pair[1]} ({min_corr:.2f})")
        
        results['insights'] = insights
        
        return results