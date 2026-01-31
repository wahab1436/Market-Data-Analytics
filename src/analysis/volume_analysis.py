"""
Volume Analysis Module
Analyzes trading volume patterns, anomalies, and price relationships
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import joblib
from pathlib import Path


class VolumeAnalysis:
    """Analyzes trading volume behavior and patterns."""
    
    def __init__(self, config: Dict[str, Any], logger):
        """Initialize volume analyzer."""
        self.config = config
        self.logger = logger
        self.color_palette = config['dashboard']['color_palette']
        self.artifacts_path = Path(config['paths']['artifacts'])
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive volume analysis."""
        results = {}
        
        by_symbol = data['by_symbol']
        
        for symbol, df in by_symbol.items():
            self.logger.info(f"Analyzing volume for {symbol}")
            
            symbol_results = {
                'charts': {},
                'insights': [],
                'metrics': {}
            }
            
            # Create volume charts
            volume_charts = self._create_volume_charts(df, symbol)
            symbol_results['charts'].update(volume_charts)
            
            # Calculate volume metrics
            metrics = self._calculate_volume_metrics(df)
            symbol_results['metrics'].update(metrics)
            
            # Generate insights
            insights = self._generate_volume_insights(df, metrics)
            symbol_results['insights'].extend(insights)
            
            # Volume-price relationship analysis
            vp_results = self._analyze_volume_price_relationship(df)
            symbol_results.update(vp_results)
            
            # Volume anomalies detection
            anomaly_results = self._detect_volume_anomalies(df)
            symbol_results.update(anomaly_results)
            
            results[symbol] = symbol_results
            
            # Save symbol results
            self._save_symbol_results(symbol, symbol_results)
        
        # Cross-symbol volume analysis
        if len(by_symbol) > 1:
            cross_results = self._analyze_cross_symbol_volume(by_symbol)
            results['cross_symbol'] = cross_results
        
        return results
    
    def _create_volume_charts(self, df: pd.DataFrame, symbol: str) -> Dict[str, go.Figure]:
        """Create volume visualization charts."""
        charts = {}
        
        # 1. Volume vs Price with Volume Profile
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.4, 0.3, 0.3],
            subplot_titles=(
                f'{symbol} Price & Volume',
                'Volume Profile by Price Level',
                'Volume-Price Correlation (20-day rolling)'
            )
        )
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['close'],
                mode='lines',
                name='Close Price',
                line=dict(color=self.color_palette['primary'], width=2),
                yaxis='y1',
                hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Volume bars (second y-axis)
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
        
        # Add rolling average volume
        volume_ma = df['volume'].rolling(window=20, min_periods=20).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=volume_ma,
                mode='lines',
                name='20-day Avg Volume',
                line=dict(color=self.color_palette['warning'], width=1.5, dash='dash'),
                yaxis='y2',
                hovertemplate='20-day Avg: %{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Volume profile (histogram by price level)
        if len(df) >= 20:
            # Create price bins
            price_min = df['close'].min()
            price_max = df['close'].max()
            price_bins = np.linspace(price_min * 0.95, price_max * 1.05, 20)
            
            # Calculate volume in each price bin
            volume_by_price = []
            price_midpoints = []
            
            for i in range(len(price_bins) - 1):
                lower = price_bins[i]
                upper = price_bins[i + 1]
                mask = (df['close'] >= lower) & (df['close'] < upper)
                volume_sum = df.loc[mask, 'volume'].sum()
                volume_by_price.append(volume_sum)
                price_midpoints.append((lower + upper) / 2)
            
            fig.add_trace(
                go.Bar(
                    x=volume_by_price,
                    y=price_midpoints,
                    orientation='h',
                    name='Volume Profile',
                    marker_color=self.color_palette['secondary'],
                    hovertemplate='Price: $%{y:.2f}<br>Total Volume: %{x:,.0f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Volume-price correlation (rolling)
        if 'volume_return_corr_20d' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['volume_return_corr_20d'],
                    mode='lines',
                    name='Volume-Return Correlation',
                    line=dict(color=self.color_palette['success'], width=2),
                    hovertemplate='Correlation: %{y:.3f}<extra></extra>'
                ),
                row=3, col=1
            )
            
            # Add significance level lines
            significance = 1.96 / np.sqrt(20)  # 95% confidence for 20-day window
            fig.add_hline(
                y=significance,
                line_width=1,
                line_dash="dash",
                line_color=self.color_palette['danger'],
                row=3, col=1
            )
            
            fig.add_hline(
                y=-significance,
                line_width=1,
                line_dash="dash",
                line_color=self.color_palette['danger'],
                row=3, col=1
            )
        
        fig.update_layout(
            height=800,
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
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Price Level ($)", row=2, col=1)
        fig.update_yaxes(title_text="Correlation", row=3, col=1)
        fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
        
        charts['volume_price_composite'] = fig
        
        # 2. Volume Anomalies Detection
        fig2 = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=('Volume Anomalies Detection', 'Volume Z-Score')
        )
        
        # Volume with anomalies highlighted
        volume = df['volume']
        
        # Calculate volume statistics
        log_volume = np.log1p(volume)
        volume_mean = log_volume.rolling(window=20, min_periods=20).mean()
        volume_std = log_volume.rolling(window=20, min_periods=20).std()
        
        # Identify anomalies (outside 2 standard deviations)
        z_scores = (log_volume - volume_mean) / (volume_std + 1e-8)
        anomalies = z_scores.abs() > 2
        
        # Normal volume days
        normal_days = ~anomalies
        fig2.add_trace(
            go.Bar(
                x=df.index[normal_days],
                y=volume[normal_days],
                name='Normal Volume',
                marker_color='rgba(46, 134, 171, 0.6)',
                hovertemplate='Volume: %{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Anomaly volume days
        if anomalies.any():
            fig2.add_trace(
                go.Bar(
                    x=df.index[anomalies],
                    y=volume[anomalies],
                    name='Volume Anomaly (>2σ)',
                    marker_color=self.color_palette['danger'],
                    hovertemplate='Volume: %{y:,.0f}<br>Z-score: %{customdata:.1f}<extra></extra>',
                    customdata=z_scores[anomalies]
                ),
                row=1, col=1
            )
        
        # Volume Z-score
        fig2.add_trace(
            go.Scatter(
                x=df.index,
                y=z_scores,
                mode='lines',
                name='Volume Z-Score',
                line=dict(color=self.color_palette['secondary'], width=1.5),
                hovertemplate='Z-score: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add anomaly threshold lines
        fig2.add_hline(
            y=2,
            line_width=1,
            line_dash="dash",
            line_color=self.color_palette['danger'],
            row=2, col=1
        )
        
        fig2.add_hline(
            y=-2,
            line_width=1,
            line_dash="dash",
            line_color=self.color_palette['danger'],
            row=2, col=1
        )
        
        fig2.add_hline(
            y=0,
            line_width=1,
            line_color=self.color_palette['text'],
            row=2, col=1
        )
        
        fig2.update_layout(
            height=600,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white',
            barmode='overlay'
        )
        
        fig2.update_xaxes(title_text="Date", row=2, col=1)
        fig2.update_yaxes(title_text="Volume", row=1, col=1)
        fig2.update_yaxes(title_text="Z-Score", row=2, col=1)
        
        charts['volume_anomalies'] = fig2
        
        # 3. Volume Breakdown by Return Type
        fig3 = go.Figure()
        
        # Categorize days by return type
        returns = df['return']
        large_up = returns > returns.quantile(0.75)
        large_down = returns < returns.quantile(0.25)
        moderate = ~(large_up | large_down)
        
        # Calculate average volume by category
        categories = {
            'Large Up Days (>75th %ile)': large_up,
            'Large Down Days (<25th %ile)': large_down,
            'Moderate Days': moderate
        }
        
        avg_volumes = []
        categories_list = []
        
        for name, mask in categories.items():
            if mask.any():
                avg_vol = volume[mask].mean()
                avg_volumes.append(avg_vol)
                categories_list.append(name)
        
        if avg_volumes:
            fig3.add_trace(
                go.Bar(
                    x=categories_list,
                    y=avg_volumes,
                    name='Average Volume',
                    marker_color=[
                        self.color_palette['success'],
                        self.color_palette['danger'],
                        self.color_palette['primary']
                    ],
                    hovertemplate='%{x}<br>Avg Volume: %{y:,.0f}<extra></extra>'
                )
            )
            
            # Add overall average line
            overall_avg = volume.mean()
            fig3.add_hline(
                y=overall_avg,
                line_width=2,
                line_dash="dash",
                line_color=self.color_palette['text'],
                annotation_text=f"Overall Average: {overall_avg:,.0f}",
                annotation_position="top right"
            )
        
        fig3.update_layout(
            title=f'{symbol} Volume by Return Category',
            xaxis_title='Return Category',
            yaxis_title='Average Volume',
            height=400,
            showlegend=False,
            template='plotly_white'
        )
        
        charts['volume_by_return_category'] = fig3
        
        return charts
    
    def _calculate_volume_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate key volume metrics."""
        metrics = {}
        
        if len(df) < 20:
            return metrics
        
        volume = df['volume']
        returns = df['return']
        
        # Basic volume statistics
        metrics['current_volume'] = float(volume.iloc[-1])
        metrics['avg_volume_20d'] = float(volume.tail(20).mean())
        metrics['volume_std_20d'] = float(volume.tail(20).std())
        
        # Volume ratios
        if metrics['avg_volume_20d'] > 0:
            metrics['volume_ratio_current_vs_avg'] = float(metrics['current_volume'] / metrics['avg_volume_20d'])
        else:
            metrics['volume_ratio_current_vs_avg'] = 1.0
            
        if metrics['avg_volume_20d'] > 0:
            metrics['volume_variability'] = float(metrics['volume_std_20d'] / metrics['avg_volume_20d'])
        else:
            metrics['volume_variability'] = 0.0
        
        # Dollar volume metrics
        if 'dollar_volume' in df.columns:
            dollar_volume = df['dollar_volume']
            metrics['current_dollar_volume'] = float(dollar_volume.iloc[-1])
            metrics['avg_dollar_volume_20d'] = float(dollar_volume.tail(20).mean())
        
        # Volume-return relationships
        if 'volume_return_corr_20d' in df.columns:
            corr_val = df['volume_return_corr_20d'].iloc[-1]
            metrics['current_volume_return_corr'] = float(corr_val) if not pd.isna(corr_val) else 0.0
        
        # Calculate overall volume-return correlation
        valid_data = returns.notna() & volume.notna()
        if valid_data.sum() > 10:
            corr = returns[valid_data].corr(volume[valid_data])
            metrics['overall_volume_return_corr'] = float(corr) if not pd.isna(corr) else 0.0
            
            # Correlation on up vs down days
            up_days = returns[valid_data] > 0
            down_days = returns[valid_data] < 0
            
            if up_days.sum() > 5:
                corr_up = returns[valid_data][up_days].corr(volume[valid_data][up_days])
                metrics['volume_return_corr_up_days'] = float(corr_up) if not pd.isna(corr_up) else 0.0
            
            if down_days.sum() > 5:
                corr_down = returns[valid_data][down_days].corr(volume[valid_data][down_days])
                metrics['volume_return_corr_down_days'] = float(corr_down) if not pd.isna(corr_down) else 0.0
        
        # Volume distribution metrics
        log_volume = np.log1p(volume)
        metrics['volume_skewness'] = float(log_volume.skew())
        metrics['volume_kurtosis'] = float(log_volume.kurtosis())
        
        # Volume percentiles
        metrics['volume_percentile_current'] = float(stats.percentileofscore(volume, metrics['current_volume']))
        metrics['volume_percentile_20d_avg'] = float(stats.percentileofscore(volume, metrics['avg_volume_20d']))
        
        # Extreme volume days
        volume_mean = volume.mean()
        volume_std = volume.std()
        extreme_high_volume = (volume > volume_mean + 2 * volume_std).sum()
        extreme_low_volume = (volume < volume_mean - 2 * volume_std).sum()
        
        metrics['extreme_high_volume_days'] = int(extreme_high_volume)
        metrics['extreme_low_volume_days'] = int(extreme_low_volume)
        metrics['pct_extreme_volume_days'] = float((extreme_high_volume + extreme_low_volume) / len(volume) * 100)
        
        return metrics
    
    def _generate_volume_insights(self, df: pd.DataFrame, metrics: Dict[str, float]) -> List[str]:
        """Generate textual insights from volume analysis."""
        insights = []
        
        if not metrics:
            return insights
        
        # Current volume level insight
        current_volume = metrics.get('current_volume', 0)
        avg_volume = metrics.get('avg_volume_20d', 0)
        volume_ratio = metrics.get('volume_ratio_current_vs_avg', 1)
        
        if volume_ratio > 1.5:
            insights.append(f"High volume day: Current volume ({current_volume:,.0f}) is {volume_ratio:.1f}x above 20-day average.")
        elif volume_ratio < 0.7:
            insights.append(f"Low volume day: Current volume ({current_volume:,.0f}) is {1/volume_ratio:.1f}x below 20-day average.")
        
        # Volume percentile insight
        volume_percentile = metrics.get('volume_percentile_current', 50)
        if volume_percentile > 80:
            insights.append(f"Volume in top {100-volume_percentile:.0f}th percentile historically.")
        elif volume_percentile < 20:
            insights.append(f"Volume in bottom {volume_percentile:.0f}th percentile historically.")
        
        # Volume-return relationship insight
        corr = metrics.get('current_volume_return_corr')
        if corr is not None:
            if abs(corr) > 0.3:
                if corr > 0:
                    insights.append(f"Strong positive volume-return relationship: High volume accompanies positive returns.")
                else:
                    insights.append(f"Strong negative volume-return relationship: High volume accompanies negative returns.")
            elif abs(corr) < 0.1:
                insights.append(f"Weak volume-return relationship: Volume changes independent of returns.")
        
        # Volume distribution insight
        volume_skew = metrics.get('volume_skewness', 0)
        if volume_skew > 1:
            insights.append(f"Volume distribution is right-skewed: Few very high volume days.")
        elif volume_skew < -1:
            insights.append(f"Volume distribution is left-skewed: Few very low volume days.")
        
        # Extreme volume insight
        extreme_days_pct = metrics.get('pct_extreme_volume_days', 0)
        if extreme_days_pct > 10:
            insights.append(f"Frequent extreme volume days: {extreme_days_pct:.1f}% of days have >2σ volume.")
        
        # Volume variability insight
        volume_variability = metrics.get('volume_variability', 0)
        if volume_variability > 0.5:
            insights.append(f"High volume variability: Daily volume changes significantly.")
        
        return insights
    
    def _analyze_volume_price_relationship(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relationship between volume and price movements."""
        results = {}
        
        if len(df) < 20:
            return results
        
        volume = df['volume']
        returns = df['return']
        close = df['close']
        
        # 1. Volume confirmation of price moves
        # Large moves on high volume are more significant
        large_moves = returns.abs() > returns.abs().quantile(0.75)
        high_volume = volume > volume.quantile(0.75)
        
        volume_confirmed_moves = (large_moves & high_volume).sum()
        volume_unconfirmed_moves = (large_moves & ~high_volume).sum()
        
        results['volume_confirmed_large_moves'] = int(volume_confirmed_moves)
        results['volume_unconfirmed_large_moves'] = int(volume_unconfirmed_moves)
        results['volume_confirmation_ratio'] = float(
            volume_confirmed_moves / (volume_confirmed_moves + volume_unconfirmed_moves + 1e-8)
        )
        
        # 2. Volume on up vs down days
        up_days = returns > 0
        down_days = returns < 0
        
        results['avg_volume_up_days'] = float(volume[up_days].mean()) if up_days.any() else 0.0
        results['avg_volume_down_days'] = float(volume[down_days].mean()) if down_days.any() else 0.0
        results['volume_up_down_ratio'] = float(
            results['avg_volume_up_days'] / (results['avg_volume_down_days'] + 1e-8)
        )
        
        # 3. Volume preceding price moves
        # Does high volume predict next day's return?
        volume_quantiles = pd.qcut(volume, q=4, labels=['Very Low', 'Low', 'High', 'Very High'])
        
        next_day_returns_by_volume = {}
        for label in ['Very Low', 'Low', 'High', 'Very High']:
            mask = volume_quantiles == label
            if mask.any() and mask.shift(-1).any():
                next_day_returns = returns.shift(-1)[mask]
                next_day_returns_by_volume[label] = {
                    'mean': float(next_day_returns.mean() * 100),
                    'std': float(next_day_returns.std() * 100),
                    'count': int(len(next_day_returns))
                }
        
        results['next_day_returns_by_volume'] = next_day_returns_by_volume
        
        # 4. Volume accumulation/distribution
        # Simple accumulation: volume * return sign
        accumulation = (volume * np.sign(returns)).cumsum()
        results['volume_accumulation_trend'] = 'Accumulation' if accumulation.iloc[-1] > 0 else 'Distribution'
        results['volume_accumulation_value'] = float(accumulation.iloc[-1])
        
        return results
    
    def _detect_volume_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect and analyze volume anomalies."""
        results = {}
        
        if len(df) < 20:
            return results
        
        volume = df['volume']
        returns = df['return']
        
        # Use statistical methods to detect anomalies
        log_volume = np.log1p(volume)
        
        # Rolling statistics
        rolling_mean = log_volume.rolling(window=20, min_periods=20).mean()
        rolling_std = log_volume.rolling(window=20, min_periods=20).std()
        
        # Calculate z-scores
        z_scores = (log_volume - rolling_mean) / (rolling_std + 1e-8)
        
        # Detect anomalies (outside 2 standard deviations)
        high_volume_anomalies = z_scores > 2
        low_volume_anomalies = z_scores < -2
        
        results['high_volume_anomaly_count'] = int(high_volume_anomalies.sum())
        results['low_volume_anomaly_count'] = int(low_volume_anomalies.sum())
        results['total_volume_anomalies'] = results['high_volume_anomaly_count'] + results['low_volume_anomaly_count']
        
        # Analyze what happens after volume anomalies
        if high_volume_anomalies.any():
            anomaly_indices = high_volume_anomalies[high_volume_anomalies].index
            
            # Returns on anomaly days
            anomaly_day_returns = returns[high_volume_anomalies]
            results['avg_return_on_high_volume_anomaly'] = float(anomaly_day_returns.mean() * 100)
            
            # Returns following anomaly days
            if len(anomaly_indices) > 0:
                next_day_returns = []
                for idx in anomaly_indices:
                    idx_pos = df.index.get_loc(idx)
                    if idx_pos + 1 < len(returns):
                        next_day_returns.append(returns.iloc[idx_pos + 1])
                
                if next_day_returns:
                    results['avg_next_day_return_after_high_volume'] = float(np.mean(next_day_returns) * 100)
                    results['pct_positive_next_day_after_high_volume'] = float(
                        sum(1 for r in next_day_returns if r > 0) / len(next_day_returns) * 100
                    )
        
        # Cluster analysis: are volume anomalies clustered?
        if results['total_volume_anomalies'] > 1:
            anomaly_mask = high_volume_anomalies | low_volume_anomalies
            anomaly_indices = np.where(anomaly_mask)[0]
            
            # Calculate average gap between anomalies
            if len(anomaly_indices) > 1:
                gaps = np.diff(anomaly_indices)
                results['avg_gap_between_anomalies'] = float(np.mean(gaps))
                results['anomaly_clustering_score'] = float(1 / np.mean(gaps))  # Higher = more clustered
            else:
                results['avg_gap_between_anomalies'] = 0.0
                results['anomaly_clustering_score'] = 0.0
        
        # Volume anomaly by price action
        anomaly_details = []
        if high_volume_anomalies.any():
            for date in high_volume_anomalies[high_volume_anomalies].index:
                vol = volume.loc[date]
                ret = returns.loc[date] * 100
                price = df['close'].loc[date]
                z = z_scores.loc[date]
                
                anomaly_details.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'volume': float(vol),
                    'return_pct': float(ret),
                    'price': float(price),
                    'z_score': float(z)
                })
        
        results['high_volume_anomaly_details'] = anomaly_details[:10]  # Limit to 10 most recent
        
        return results
    
    def _analyze_cross_symbol_volume(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze volume relationships between symbols."""
        results = {}
        
        symbols = list(data.keys())
        if len(symbols) < 2:
            return results
        
        # Calculate normalized volume for each symbol
        volume_data = {}
        for symbol, df in data.items():
            volume = df['volume']
            # Normalize by 20-day moving average
            volume_ma = volume.rolling(20, min_periods=20).mean()
            normalized_volume = volume / volume_ma
            volume_data[symbol] = normalized_volume.dropna()
        
        # Align volume series
        vol_df = pd.DataFrame(volume_data).dropna()
        
        if len(vol_df) < 20:
            return results
        
        # Correlation of normalized volumes
        vol_correlation = vol_df.corr()
        
        results['normalized_volume_correlation'] = vol_correlation.to_dict()
        
        # Create volume correlation heatmap
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
            title='Normalized Volume Correlation (vs 20-day MA)',
            height=400,
            width=500,
            template='plotly_white'
        )
        
        results['volume_correlation_chart'] = fig
        
        # Volume leadership analysis
        insights = []
        
        if len(symbols) >= 2:
            # Find which symbol's volume leads others
            lead_lag_correlations = {}
            
            for i, sym1 in enumerate(symbols):
                for sym2 in symbols[i+1:]:
                    if sym1 in vol_df.columns and sym2 in vol_df.columns:
                        # Cross-correlation at different lags
                        max_lag = 5
                        best_corr = -1
                        best_lag = 0
                        
                        for lag in range(-max_lag, max_lag + 1):
                            if lag < 0:
                                corr = vol_df[sym1].corr(vol_df[sym2].shift(-lag))
                            elif lag > 0:
                                corr = vol_df[sym1].shift(lag).corr(vol_df[sym2])
                            else:
                                corr = vol_df[sym1].corr(vol_df[sym2])
                            
                            if not pd.isna(corr) and abs(corr) > abs(best_corr):
                                best_corr = corr
                                best_lag = lag
                        
                        lead_lag_correlations[f'{sym1}_{sym2}'] = {
                            'best_correlation': float(best_corr),
                            'best_lag': int(best_lag),
                            'interpretation': f'{sym1} leads by {-best_lag} days' if best_lag < 0 else
                                            f'{sym2} leads by {best_lag} days' if best_lag > 0 else
                                            'No lead-lag relationship'
                        }
            
            results['volume_lead_lag_analysis'] = lead_lag_correlations
            
            # Generate insights
            for pair, analysis in lead_lag_correlations.items():
                sym1, sym2 = pair.split('_')
                lag = analysis['best_lag']
                corr = analysis['best_correlation']
                
                if abs(lag) >= 2 and abs(corr) > 0.3:
                    if lag < 0:
                        insights.append(f"{sym1} volume changes lead {sym2} by {-lag} days (corr: {corr:.2f})")
                    else:
                        insights.append(f"{sym2} volume changes lead {sym1} by {lag} days (corr: {corr:.2f})")
        
        results['insights'] = insights
        
        return results
    
    def _save_symbol_results(self, symbol: str, results: Dict[str, Any]) -> None:
        """Save volume analysis results for a symbol to artifacts folder."""
        try:
            # Create artifacts directory if it doesn't exist
            self.artifacts_path.mkdir(parents=True, exist_ok=True)
            
            # Save results to pickle file
            output_file = self.artifacts_path / f"{symbol}_volume_analysis.pkl"
            joblib.dump(results, output_file)
            
            self.logger.info(f"Saved volume analysis results for {symbol} to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving volume analysis results for {symbol}: {e}")