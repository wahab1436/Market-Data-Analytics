"""
Volatility Analysis Module
Analyzes market volatility patterns, regimes, and extremes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import joblib
from pathlib import Path


class VolatilityAnalysis:
    """Analyzes volatility behavior, regimes, and patterns."""
    
    def __init__(self, config: Dict[str, Any], logger):
        """Initialize volatility analyzer."""
        self.config = config
        self.logger = logger
        self.color_palette = config['dashboard']['color_palette']
        self.artifacts_path = Path(config['paths']['artifacts'])
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive volatility analysis."""
        results = {}
        
        by_symbol = data['by_symbol']
        
        for symbol, df in by_symbol.items():
            self.logger.info(f"Analyzing volatility for {symbol}")
            
            symbol_results = {
                'charts': {},
                'insights': [],
                'metrics': {}
            }
            
            # Create volatility charts
            volatility_charts = self._create_volatility_charts(df, symbol)
            symbol_results['charts'].update(volatility_charts)
            
            # Calculate volatility metrics
            metrics = self._calculate_volatility_metrics(df)
            symbol_results['metrics'].update(metrics)
            
            # Generate insights
            insights = self._generate_volatility_insights(df, metrics)
            symbol_results['insights'].extend(insights)
            
            # Regime analysis
            regime_results = self._analyze_volatility_regimes(df)
            symbol_results.update(regime_results)
            
            # Extreme moves analysis
            extreme_results = self._analyze_extreme_moves(df)
            symbol_results.update(extreme_results)
            
            results[symbol] = symbol_results
            
            # Save symbol results
            self._save_symbol_results(symbol, symbol_results)
        
        # Cross-symbol volatility analysis
        if len(by_symbol) > 1:
            cross_results = self._analyze_cross_symbol_volatility(by_symbol)
            results['cross_symbol'] = cross_results
        
        return results
    
    def _create_volatility_charts(self, df: pd.DataFrame, symbol: str) -> Dict[str, go.Figure]:
        """Create volatility visualization charts."""
        charts = {}
        
        # 1. Volatility Timeline with Returns
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.5],
            subplot_titles=(f'{symbol} Rolling Volatility', 'Daily Returns with Volatility Regime')
        )
        
        # Calculate rolling volatility (20-day)
        rolling_vol = df['return'].rolling(window=20, min_periods=20).std() * np.sqrt(252) * 100
        
        # Volatility line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=rolling_vol,
                mode='lines',
                name='20-Day Annualized Vol (%)',
                line=dict(color=self.color_palette['primary'], width=2),
                fill='tozeroy',
                fillcolor='rgba(46, 134, 171, 0.1)',
                hovertemplate='Date: %{x}<br>Volatility: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add volatility bands (mean +/- 1 std)
        vol_mean = rolling_vol.mean()
        vol_std = rolling_vol.std()
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[vol_mean] * len(df),
                mode='lines',
                name='Mean Volatility',
                line=dict(color=self.color_palette['text'], width=1, dash='dash'),
                hovertemplate='Mean: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[vol_mean + vol_std] * len(df),
                mode='lines',
                name='+1 Std Dev',
                line=dict(color=self.color_palette['warning'], width=1, dash='dot'),
                hovertemplate='+1σ: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[vol_mean - vol_std] * len(df),
                mode='lines',
                name='-1 Std Dev',
                line=dict(color=self.color_palette['success'], width=1, dash='dot'),
                hovertemplate='-1σ: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Daily returns with volatility coloring
        volatility_zones = self._calculate_volatility_zones(rolling_vol, vol_mean, vol_std)
        
        colors = []
        for zone in volatility_zones:
            if zone == 'High':
                colors.append(self.color_palette['danger'])
            elif zone == 'Low':
                colors.append(self.color_palette['success'])
            else:
                colors.append(self.color_palette['secondary'])
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['return'] * 100,
                name='Daily Return %',
                marker_color=colors,
                hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<br>Vol Regime: %{customdata}',
                customdata=volatility_zones
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)
        
        charts['volatility_timeline'] = fig
        
        # 2. Volatility Distribution
        fig2 = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Volatility Distribution', 'Volatility Autocorrelation'),
            column_widths=[0.6, 0.4]
        )
        
        # Histogram of volatility
        fig2.add_trace(
            go.Histogram(
                x=rolling_vol.dropna(),
                nbinsx=40,
                name='Volatility Distribution',
                marker_color=self.color_palette['primary'],
                opacity=0.7,
                hovertemplate='Volatility: %{x:.1f}%<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add vertical lines for statistics
        fig2.add_vline(
            x=vol_mean,
            line_width=2,
            line_dash="dash",
            line_color=self.color_palette['text'],
            row=1, col=1
        )
        
        fig2.add_vline(
            x=vol_mean + vol_std,
            line_width=1,
            line_dash="dot",
            line_color=self.color_palette['warning'],
            row=1, col=1
        )
        
        fig2.add_vline(
            x=vol_mean - vol_std,
            line_width=1,
            line_dash="dot",
            line_color=self.color_palette['success'],
            row=1, col=1
        )
        
        # Volatility autocorrelation
        max_lags = 30
        autocorr = [rolling_vol.autocorr(lag=i) for i in range(1, max_lags + 1)]
        
        fig2.add_trace(
            go.Bar(
                x=list(range(1, max_lags + 1)),
                y=autocorr,
                name='Autocorrelation',
                marker_color=self.color_palette['secondary'],
                hovertemplate='Lag: %{x}<br>Autocorr: %{y:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Add significance bands (95% confidence)
        significance_level = 1.96 / np.sqrt(len(rolling_vol.dropna()))
        fig2.add_hline(
            y=significance_level,
            line_width=1,
            line_dash="dash",
            line_color=self.color_palette['danger'],
            row=1, col=2
        )
        
        fig2.add_hline(
            y=-significance_level,
            line_width=1,
            line_dash="dash",
            line_color=self.color_palette['danger'],
            row=1, col=2
        )
        
        fig2.update_layout(
            height=400,
            showlegend=False,
            template='plotly_white'
        )
        
        fig2.update_xaxes(title_text="Volatility (%)", row=1, col=1)
        fig2.update_yaxes(title_text="Frequency", row=1, col=1)
        fig2.update_xaxes(title_text="Lag (days)", row=1, col=2)
        fig2.update_yaxes(title_text="Autocorrelation", row=1, col=2)
        
        charts['volatility_distribution'] = fig2
        
        # 3. Volatility Clustering Chart
        fig3 = go.Figure()
        
        # Absolute returns as proxy for volatility
        abs_returns = df['return'].abs() * 100
        
        # Scatter plot showing volatility clustering
        fig3.add_trace(
            go.Scatter(
                x=df.index,
                y=abs_returns,
                mode='markers',
                name='Absolute Daily Return (%)',
                marker=dict(
                    size=6,
                    color=abs_returns,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Absolute Return %")
                ),
                hovertemplate='Date: %{x}<br>Abs Return: %{y:.2f}%<extra></extra>'
            )
        )
        
        # Add rolling average of absolute returns
        rolling_abs = abs_returns.rolling(window=20, min_periods=20).mean()
        fig3.add_trace(
            go.Scatter(
                x=df.index,
                y=rolling_abs,
                mode='lines',
                name='20-day Average',
                line=dict(color=self.color_palette['warning'], width=2),
                hovertemplate='20-day Avg: %{y:.2f}%<extra></extra>'
            )
        )
        
        fig3.update_layout(
            title=f'{symbol} Volatility Clustering (Absolute Returns)',
            xaxis_title='Date',
            yaxis_title='Absolute Daily Return (%)',
            height=400,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        charts['volatility_clustering'] = fig3
        
        return charts
    
    def _calculate_volatility_zones(self, rolling_vol: pd.Series, mean: float, std: float) -> List[str]:
        """Categorize volatility into zones."""
        zones = []
        for vol in rolling_vol:
            if pd.isna(vol):
                zones.append('Unknown')
            elif vol > mean + std:
                zones.append('High')
            elif vol < mean - std:
                zones.append('Low')
            else:
                zones.append('Normal')
        return zones
    
    def _calculate_volatility_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate key volatility metrics."""
        metrics = {}
        
        if len(df) < 20:
            return metrics
        
        # Basic volatility metrics
        returns = df['return'].dropna()
        metrics['current_volatility_20d'] = float(returns.tail(20).std() * np.sqrt(252) * 100)
        metrics['avg_volatility_20d'] = float(returns.rolling(20).std().dropna().mean() * np.sqrt(252) * 100)
        metrics['volatility_std'] = float(returns.rolling(20).std().dropna().std() * np.sqrt(252) * 100)
        
        # Volatility percentiles
        rolling_vol = returns.rolling(20).std().dropna() * np.sqrt(252) * 100
        metrics['volatility_percentile'] = float(stats.percentileofscore(rolling_vol, metrics['current_volatility_20d']))
        
        # Volatility ratios
        if metrics['avg_volatility_20d'] > 0:
            metrics['volatility_ratio_current_vs_avg'] = float(metrics['current_volatility_20d'] / metrics['avg_volatility_20d'])
        else:
            metrics['volatility_ratio_current_vs_avg'] = 1.0
        
        # Extreme volatility metrics
        metrics['days_above_2std'] = int((returns.abs() > (2 * returns.std())).sum())
        metrics['pct_days_above_2std'] = float(metrics['days_above_2std'] / len(returns) * 100)
        
        # Maximum volatility periods
        rolling_30d_vol = returns.rolling(30).std().dropna() * np.sqrt(252) * 100
        if len(rolling_30d_vol) > 0:
            metrics['max_30d_volatility'] = float(rolling_30d_vol.max())
            metrics['min_30d_volatility'] = float(rolling_30d_vol.min())
            metrics['volatility_range_30d'] = float(metrics['max_30d_volatility'] - metrics['min_30d_volatility'])
        
        # Volatility persistence (autocorrelation)
        autocorr_1 = returns.autocorr(lag=1)
        autocorr_5 = returns.autocorr(lag=5)
        metrics['volatility_persistence_1d'] = float(autocorr_1) if not pd.isna(autocorr_1) else 0.0
        metrics['volatility_persistence_5d'] = float(autocorr_5) if not pd.isna(autocorr_5) else 0.0
        
        return metrics
    
    def _generate_volatility_insights(self, df: pd.DataFrame, metrics: Dict[str, float]) -> List[str]:
        """Generate textual insights from volatility analysis."""
        insights = []
        
        if not metrics:
            return insights
        
        # Current volatility level insight
        current_vol = metrics.get('current_volatility_20d', 0)
        avg_vol = metrics.get('avg_volatility_20d', 0)
        vol_ratio = metrics.get('volatility_ratio_current_vs_avg', 1)
        
        if vol_ratio > 1.5:
            insights.append(f"High volatility regime: Current volatility ({current_vol:.1f}%) is {vol_ratio:.1f}x above average.")
        elif vol_ratio < 0.7:
            insights.append(f"Low volatility regime: Current volatility ({current_vol:.1f}%) is {1/vol_ratio:.1f}x below average.")
        else:
            insights.append(f"Normal volatility regime: Current volatility ({current_vol:.1f}%) near historical average.")
        
        # Volatility percentile insight
        percentile = metrics.get('volatility_percentile', 50)
        if percentile > 75:
            insights.append(f"Volatility in top {100-percentile:.0f}th percentile historically.")
        elif percentile < 25:
            insights.append(f"Volatility in bottom {percentile:.0f}th percentile historically.")
        
        # Extreme moves insight
        extreme_days_pct = metrics.get('pct_days_above_2std', 0)
        if extreme_days_pct > 5:
            insights.append(f"Elevated extreme moves: {extreme_days_pct:.1f}% of days had >2σ returns (normal: ~5%).")
        elif extreme_days_pct < 2:
            insights.append(f"Few extreme moves: Only {extreme_days_pct:.1f}% of days had >2σ returns.")
        
        # Volatility persistence insight
        persistence = metrics.get('volatility_persistence_1d', 0)
        if abs(persistence) > 0.1:
            if persistence > 0:
                insights.append(f"Volatility clustering detected: High autocorrelation ({persistence:.2f}) suggests persistent volatility.")
            else:
                insights.append(f"Mean reversion pattern: Negative autocorrelation ({persistence:.2f}) suggests volatility alternation.")
        
        # Volatility range insight
        vol_range = metrics.get('volatility_range_30d', 0)
        if vol_range > 20:
            insights.append(f"Wide volatility swings: 30-day range of {vol_range:.1f}% indicates changing market conditions.")
        
        return insights
    
    def _analyze_volatility_regimes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze different volatility regimes."""
        results = {}
        
        if len(df) < 60:
            return results
        
        returns = df['return'].dropna()
        
        # Use Hidden Markov Model-like simple regime detection
        # Simple threshold-based approach for MVP
        rolling_vol = returns.rolling(20).std().dropna() * np.sqrt(252) * 100
        
        if len(rolling_vol) == 0:
            return results
        
        vol_mean = rolling_vol.mean()
        vol_std = rolling_vol.std()
        
        # Classify regimes
        high_vol_mask = rolling_vol > vol_mean + vol_std
        low_vol_mask = rolling_vol < vol_mean - vol_std
        normal_vol_mask = ~(high_vol_mask | low_vol_mask)
        
        # Count days in each regime
        results['high_vol_days'] = int(high_vol_mask.sum())
        results['low_vol_days'] = int(low_vol_mask.sum())
        results['normal_vol_days'] = int(normal_vol_mask.sum())
        results['pct_high_vol'] = float(results['high_vol_days'] / len(rolling_vol) * 100)
        results['pct_low_vol'] = float(results['low_vol_days'] / len(rolling_vol) * 100)
        
        # Average returns by regime
        regime_returns = {
            'high': float(returns[high_vol_mask.index[high_vol_mask]].mean()) if high_vol_mask.any() else 0.0,
            'low': float(returns[low_vol_mask.index[low_vol_mask]].mean()) if low_vol_mask.any() else 0.0,
            'normal': float(returns[normal_vol_mask.index[normal_vol_mask]].mean()) if normal_vol_mask.any() else 0.0
        }
        results['avg_return_by_regime'] = regime_returns
        
        # Current regime
        if rolling_vol.iloc[-1] > vol_mean + vol_std:
            results['current_regime'] = 'High Volatility'
        elif rolling_vol.iloc[-1] < vol_mean - vol_std:
            results['current_regime'] = 'Low Volatility'
        else:
            results['current_regime'] = 'Normal Volatility'
        
        # Regime duration analysis
        results['avg_regime_duration'] = self._calculate_avg_regime_duration(
            high_vol_mask, low_vol_mask, normal_vol_mask
        )
        
        return results
    
    def _calculate_avg_regime_duration(self, *masks: pd.Series) -> Dict[str, float]:
        """Calculate average duration of each regime."""
        durations = {}
        
        regime_names = ['high', 'low', 'normal']
        
        for name, mask in zip(regime_names, masks):
            if mask.empty or not mask.any():
                durations[f'avg_{name}_duration'] = 0.0
                continue
            
            # Find consecutive True values
            mask_diff = mask.astype(int).diff()
            start_indices = mask_diff[mask_diff == 1].index
            end_indices = mask_diff[mask_diff == -1].index
            
            # Handle edge cases
            if mask.iloc[0]:
                start_indices = start_indices.insert(0, mask.index[0])
            if mask.iloc[-1]:
                end_indices = end_indices.insert(len(end_indices), mask.index[-1])
            
            if len(start_indices) != len(end_indices):
                # Should not happen with proper handling
                durations[f'avg_{name}_duration'] = 0.0
                continue
            
            regime_lengths = []
            for start, end in zip(start_indices, end_indices):
                regime_mask = mask.loc[start:end]
                length = len(regime_mask[regime_mask])
                regime_lengths.append(length)
            
            if regime_lengths:
                durations[f'avg_{name}_duration'] = float(np.mean(regime_lengths))
            else:
                durations[f'avg_{name}_duration'] = 0.0
        
        return durations
    
    def _analyze_extreme_moves(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze extreme price movements."""
        results = {}
        
        if len(df) < 10:
            return results
        
        returns = df['return'].dropna()
        
        # Define extreme moves (outside 2 standard deviations)
        std_dev = returns.std()
        extreme_threshold = 2 * std_dev
        
        # Identify extreme moves
        extreme_positive = returns[returns > extreme_threshold]
        extreme_negative = returns[returns < -extreme_threshold]
        
        results['extreme_positive_count'] = int(len(extreme_positive))
        results['extreme_negative_count'] = int(len(extreme_negative))
        results['total_extreme_moves'] = results['extreme_positive_count'] + results['extreme_negative_count']
        results['extreme_move_pct'] = float(results['total_extreme_moves'] / len(returns) * 100)
        
        # Statistics of extreme moves
        if len(extreme_positive) > 0:
            results['avg_extreme_positive'] = float(extreme_positive.mean() * 100)
            results['max_extreme_positive'] = float(extreme_positive.max() * 100)
        
        if len(extreme_negative) > 0:
            results['avg_extreme_negative'] = float(extreme_negative.mean() * 100)
            results['max_extreme_negative'] = float(extreme_negative.min() * 100)
        
        # Clustering of extreme moves
        extreme_mask = (returns.abs() > extreme_threshold)
        results['extreme_clustering_score'] = self._calculate_extreme_clustering(extreme_mask)
        
        # Recovery analysis after extreme moves
        recovery_stats = self._analyze_extreme_recovery(returns, extreme_mask)
        results.update(recovery_stats)
        
        return results
    
    def _calculate_extreme_clustering(self, extreme_mask: pd.Series) -> float:
        """Calculate how clustered extreme moves are."""
        if not extreme_mask.any():
            return 0.0
        
        # Simple clustering measure: average gap between extreme events
        extreme_indices = np.where(extreme_mask)[0]
        
        if len(extreme_indices) < 2:
            return 0.0
        
        gaps = np.diff(extreme_indices)
        avg_gap = np.mean(gaps)
        
        # Normalize by expected gap if events were random
        p_extreme = extreme_mask.mean()
        expected_gap = 1 / p_extreme if p_extreme > 0 else 0
        
        if expected_gap > 0:
            clustering_score = avg_gap / expected_gap
        else:
            clustering_score = 0.0
        
        return float(clustering_score)
    
    def _analyze_extreme_recovery(self, returns: pd.Series, extreme_mask: pd.Series) -> Dict[str, Any]:
        """Analyze how prices recover after extreme moves."""
        results = {}
        
        extreme_indices = np.where(extreme_mask)[0]
        
        if len(extreme_indices) < 5:
            return results
        
        recovery_windows = [1, 3, 5, 10]  # Days after extreme move
        
        for window in recovery_windows:
            recovery_returns = []
            for idx in extreme_indices:
                if idx + window < len(returns):
                    recovery = returns.iloc[idx + 1:idx + window + 1].sum()
                    recovery_returns.append(recovery)
            
            if recovery_returns:
                results[f'avg_recovery_{window}d'] = float(np.mean(recovery_returns) * 100)
                results[f'recovery_positive_pct_{window}d'] = float(
                    sum(1 for r in recovery_returns if r > 0) / len(recovery_returns) * 100
                )
        
        return results
    
    def _analyze_cross_symbol_volatility(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze volatility relationships between symbols."""
        results = {}
        
        symbols = list(data.keys())
        if len(symbols) < 2:
            return results
        
        # Calculate rolling volatility for each symbol
        vol_data = {}
        for symbol, df in data.items():
            returns = df['return'].dropna()
            rolling_vol = returns.rolling(20, min_periods=20).std() * np.sqrt(252) * 100
            vol_data[symbol] = rolling_vol.dropna()
        
        # Align volatility series
        vol_df = pd.DataFrame(vol_data).dropna()
        
        if len(vol_df) < 20:
            return results
        
        # Correlation of volatilities
        vol_correlation = vol_df.corr()
        
        results['volatility_correlation_matrix'] = vol_correlation.to_dict()
        
        # Create volatility correlation heatmap
        fig = go.Figure(data=go.Heatmap(
            z=vol_correlation.values,
            x=vol_correlation.columns,
            y=vol_correlation.index,
            colorscale='RdBu',
            zmid=0,
            text=vol_correlation.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Volatility Correlation Matrix (20-day rolling)',
            height=400,
            width=500,
            template='plotly_white'
        )
        
        results['volatility_correlation_chart'] = fig
        
        # Volatility lead-lag analysis
        insights = []
        
        if len(symbols) >= 2:
            # Find which symbol's volatility leads others
            lead_lag_results = {}
            for i, sym1 in enumerate(symbols):
                for sym2 in symbols[i+1:]:
                    if sym1 in vol_df.columns and sym2 in vol_df.columns:
                        # Simple cross-correlation at lag 0, 1, -1
                        corr_0 = vol_df[sym1].corr(vol_df[sym2])
                        corr_1 = vol_df[sym1].shift(1).corr(vol_df[sym2])
                        corr_minus1 = vol_df[sym1].corr(vol_df[sym2].shift(1))
                        
                        lead_lag_results[f'{sym1}_{sym2}'] = {
                            'corr_0': float(corr_0),
                            'corr_1': float(corr_1),
                            'corr_minus1': float(corr_minus1),
                            'sym1_leads' if corr_1 > corr_minus1 else 'sym2_leads': 
                                float(max(corr_1, corr_minus1))
                        }
            
            results['volatility_lead_lag'] = lead_lag_results
            
            # Generate insights
            for pair, stats in lead_lag_results.items():
                sym1, sym2 = pair.split('_')
                if stats['corr_1'] > stats['corr_minus1']:
                    insights.append(f"{sym1} volatility leads {sym2} (cross-corr: {stats['corr_1']:.2f})")
                else:
                    insights.append(f"{sym2} volatility leads {sym1} (cross-corr: {stats['corr_minus1']:.2f})")
        
        results['insights'] = insights
        
        return results
    
    def _save_symbol_results(self, symbol: str, results: Dict[str, Any]) -> None:
        """Save volatility analysis results for a symbol to artifacts folder."""
        try:
            # Create artifacts directory if it doesn't exist
            self.artifacts_path.mkdir(parents=True, exist_ok=True)
            
            # Save results to pickle file
            output_file = self.artifacts_path / f"{symbol}_volatility_analysis.pkl"
            joblib.dump(results, output_file)
            
            self.logger.info(f"Saved volatility analysis results for {symbol} to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving volatility analysis results for {symbol}: {e}")