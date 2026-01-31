"""
Similarity Analysis Module
Analyzes historical patterns, analogs, and market regimes using similarity metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import cdist
from scipy import stats
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


class SimilarityAnalysis:
    """Analyzes historical patterns and similarity between time periods."""
    
    def __init__(self, config: Dict[str, Any], logger):
        """Initialize similarity analyzer."""
        self.config = config
        self.logger = logger
        self.color_palette = config['dashboard']['color_palette']
        # Ensure path is converted to a Path object for proper handling
        self.artifacts_path = Path(config['paths']['artifacts'])
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive similarity analysis."""
        results = {}
        
        by_symbol = data['by_symbol']
        
        for symbol, df in by_symbol.items():
            self.logger.info(f"Analyzing similarity patterns for {symbol}")
            
            symbol_results = {
                'charts': {},
                'insights': [],
                'metrics': {}
            }
            
            # Create similarity charts
            similarity_charts = self._create_similarity_charts(df, symbol)
            symbol_results['charts'].update(similarity_charts)
            
            # Calculate similarity metrics
            metrics = self._calculate_similarity_metrics(df)
            symbol_results['metrics'].update(metrics)
            
            # Generate insights
            insights = self._generate_similarity_insights(df, metrics)
            symbol_results['insights'].extend(insights)
            
            # Historical analogs analysis
            analog_results = self._find_historical_analogs(df)
            symbol_results.update(analog_results)
            
            # Pattern recognition
            pattern_results = self._analyze_recurring_patterns(df)
            symbol_results.update(pattern_results)
            
            results[symbol] = symbol_results
            
            # Save symbol results using the new merge logic
            self._save_symbol_results(symbol, symbol_results)
        
        # Cross-symbol similarity analysis
        if len(by_symbol) > 1:
            cross_results = self._analyze_cross_symbol_similarity(by_symbol)
            results['cross_symbol'] = cross_results
        
        return results
    
    def _create_similarity_charts(self, df: pd.DataFrame, symbol: str) -> Dict[str, go.Figure]:
        """Create similarity visualization charts."""
        charts = {}
        
        # 1. Historical Analogs - Most Similar Periods
        analogs = self._find_top_historical_analogs(df, n_analogs=3)
        
        if analogs:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                row_heights=[0.6, 0.4],
                subplot_titles=(
                    f'{symbol} Current Pattern vs Historical Analogs',
                    'Forward Returns Following Historical Analogs'
                )
            )
            
            # Current pattern (last 20 days)
            lookback = 20
            current_dates = df.index[-lookback:]
            current_prices = df['close'].iloc[-lookback:]
            current_normalized = self._normalize_series(current_prices)
            
            # Plot current pattern
            fig.add_trace(
                go.Scatter(
                    x=list(range(lookback)),
                    y=current_normalized,
                    mode='lines+markers',
                    name='Current Pattern',
                    line=dict(color=self.color_palette['primary'], width=3),
                    marker=dict(size=6),
                    hovertemplate='Day: %{x}<br>Normalized Price: %{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Plot historical analogs
            colors = [self.color_palette['secondary'], 
                     self.color_palette['success'], 
                     self.color_palette['warning']]
            
            for i, (similarity, analog_dates) in enumerate(analogs[:3]):
                analog_prices = df.loc[analog_dates, 'close']
                analog_normalized = self._normalize_series(analog_prices)
                
                fig.add_trace(
                    go.Scatter(
                        x=list(range(lookback)),
                        y=analog_normalized,
                        mode='lines',
                        name=f'Analog {i+1} (Sim: {similarity:.2f})',
                        line=dict(color=colors[i], width=2, dash='dash'),
                        hovertemplate=f'Analog {i+1}<br>Date: {analog_dates[0].strftime("%Y-%m-%d")}<br>Similarity: {similarity:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # Forward returns following analogs
            forward_returns_data = []
            analog_labels = []
            
            for i, (similarity, analog_dates) in enumerate(analogs[:3]):
                # Get forward returns for each analog
                forward_returns = self._get_forward_returns(df, analog_dates[-1], window=20)
                forward_returns_data.append(forward_returns)
                analog_labels.append(f'Analog {i+1}')
                
                # Plot forward returns
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(forward_returns))),
                        y=forward_returns * 100,  # Convert to percentage
                        mode='lines',
                        name=f'Analog {i+1} Forward Returns',
                        line=dict(color=colors[i], width=2),
                        showlegend=False,
                        hovertemplate='Days Forward: %{x}<br>Return: %{y:.2f}%<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            # Add current forward returns (projection based on analogs)
            if forward_returns_data:
                avg_forward_returns = np.mean(forward_returns_data, axis=0)
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(avg_forward_returns))),
                        y=avg_forward_returns * 100,
                        mode='lines',
                        name='Average Forward Return (Based on Analogs)',
                        line=dict(color=self.color_palette['primary'], width=3, dash='dot'),
                        hovertemplate='Projected Return: %{y:.2f}%<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                height=700,
                showlegend=True,
                hovermode='x unified',
                template='plotly_white',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            fig.update_xaxes(title_text="Trading Days", row=1, col=1)
            fig.update_xaxes(title_text="Days After Pattern", row=2, col=1)
            fig.update_yaxes(title_text="Normalized Price", row=1, col=1)
            fig.update_yaxes(title_text="Forward Return (%)", row=2, col=1)
            
            charts['historical_analogs'] = fig
        
        # 2. Similarity Matrix Heatmap
        if len(df) >= 60:
            fig2 = self._create_similarity_matrix(df, symbol)
            charts['similarity_matrix'] = fig2
        
        # 3. Pattern Distribution and Clusters
        fig3 = self._create_pattern_cluster_chart(df, symbol)
        charts['pattern_clusters'] = fig3
        
        return charts
    
    def _normalize_series(self, series: pd.Series) -> pd.Series:
        """Normalize a price series to [0, 1] range."""
        if len(series) == 0:
            return series
        
        min_val = series.min()
        max_val = series.max()
        
        if max_val == min_val:
            return pd.Series([0.5] * len(series), index=series.index)
        
        return (series - min_val) / (max_val - min_val)
    
    def _find_top_historical_analogs(self, df: pd.DataFrame, lookback: int = 20, 
                                   n_analogs: int = 5) -> List[Tuple[float, pd.DatetimeIndex]]:
        """Find top historical analogs to current pattern."""
        if len(df) < lookback * 2:
            return []
        
        # Current pattern (last lookback days)
        current_prices = df['close'].iloc[-lookback:].values
        current_normalized = self._normalize_series(df['close'].iloc[-lookback:]).values
        
        # Search through history (exclude recent lookback)
        search_end = len(df) - lookback - 1
        analogs = []
        
        for i in range(lookback, search_end):
            historical_prices = df['close'].iloc[i-lookback:i].values
            historical_normalized = self._normalize_series(df['close'].iloc[i-lookback:i]).values
            
            # Calculate similarity (1 - normalized Euclidean distance)
            distance = np.sqrt(np.mean((current_normalized - historical_normalized) ** 2))
            similarity = 1 - distance
            
            # Also consider return pattern similarity
            current_returns = np.diff(current_prices) / current_prices[:-1]
            historical_returns = np.diff(historical_prices) / historical_prices[:-1]
            
            return_correlation = np.corrcoef(current_returns, historical_returns)[0, 1]
            
            if not np.isnan(return_correlation):
                # Combined similarity score
                combined_similarity = (similarity + (return_correlation + 1) / 2) / 2
            else:
                combined_similarity = similarity
            
            dates = df.index[i-lookback:i]
            analogs.append((combined_similarity, dates))
        
        # Sort by similarity and return top n
        analogs.sort(key=lambda x: x[0], reverse=True)
        return analogs[:n_analogs]
    
    def _get_forward_returns(self, df: pd.DataFrame, start_date: pd.Timestamp, 
                            window: int = 20) -> np.ndarray:
        """Get forward returns starting from a specific date."""
        start_idx = df.index.get_loc(start_date)
        
        if start_idx + window + 1 >= len(df):
            return np.array([])
        
        prices = df['close'].iloc[start_idx:start_idx + window + 1].values
        forward_returns = (prices[1:] / prices[0]) - 1
        
        return forward_returns
    
    def _create_similarity_matrix(self, df: pd.DataFrame, symbol: str) -> go.Figure:
        """Create a similarity matrix heatmap."""
        lookback = 20
        n_periods = min(50, len(df) // lookback)  # Limit to 50 periods for performance
        
        if n_periods < 2:
            return go.Figure()
        
        # Extract overlapping periods
        periods = []
        period_labels = []
        
        for i in range(0, n_periods * lookback, lookback):
            if i + lookback <= len(df):
                period = df['close'].iloc[i:i + lookback]
                periods.append(self._normalize_series(period).values)
                period_labels.append(df.index[i].strftime('%Y-%m'))
        
        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(periods), len(periods)))
        
        for i in range(len(periods)):
            for j in range(i, len(periods)):
                if i == j:
                    similarity = 1.0
                else:
                    # Euclidean distance between normalized patterns
                    distance = np.sqrt(np.mean((periods[i] - periods[j]) ** 2))
                    similarity = 1 - distance
                
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=period_labels,
            y=period_labels,
            colorscale='RdBu',
            zmid=0.5,
            text=np.round(similarity_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 8},
            hoverongaps=False,
            hoverinfo='text',
            hovertemplate='Period 1: %{y}<br>Period 2: %{x}<br>Similarity: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'{symbol} Pattern Similarity Matrix (20-day periods)',
            height=600,
            width=700,
            template='plotly_white',
            xaxis_title="End Date of Period",
            yaxis_title="Start Date of Period"
        )
        
        return fig
    
    def _create_pattern_cluster_chart(self, df: pd.DataFrame, symbol: str) -> go.Figure:
        """Create chart showing pattern clusters in reduced dimensions."""
        from sklearn.decomposition import PCA
        
        lookback = 20
        n_patterns = min(100, len(df) // lookback)
        
        if n_patterns < 3:
            return go.Figure()
        
        # Extract patterns
        patterns = []
        pattern_dates = []
        
        for i in range(0, n_patterns * lookback, lookback):
            if i + lookback <= len(df):
                pattern = df['close'].iloc[i:i + lookback]
                patterns.append(self._normalize_series(pattern).values)
                pattern_dates.append(df.index[i])
        
        patterns_array = np.array(patterns)
        
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        patterns_2d = pca.fit_transform(patterns_array)
        
        # Calculate return following each pattern
        forward_returns = []
        for date in pattern_dates:
            returns = self._get_forward_returns(df, date, window=10)
            if len(returns) > 0:
                forward_returns.append(returns.mean() * 100)  # Average 10-day return in %
            else:
                forward_returns.append(0)
        
        # Create scatter plot
        fig = go.Figure()
        
        # Color by forward return
        fig.add_trace(
            go.Scatter(
                x=patterns_2d[:, 0],
                y=patterns_2d[:, 1],
                mode='markers',
                marker=dict(
                    size=10,
                    color=forward_returns,
                    colorscale='RdBu',
                    colorbar=dict(title="Avg 10-day Return %"),
                    showscale=True,
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                text=[f"Start: {d.strftime('%Y-%m-%d')}<br>Return: {r:.1f}%" 
                      for d, r in zip(pattern_dates, forward_returns)],
                hovertemplate='%{text}<extra></extra>',
                name='Patterns'
            )
        )
        
        # Highlight current pattern
        if pattern_dates:
            current_idx = -1  # Most recent pattern
            fig.add_trace(
                go.Scatter(
                    x=[patterns_2d[current_idx, 0]],
                    y=[patterns_2d[current_idx, 1]],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=self.color_palette['primary'],
                        symbol='star',
                        line=dict(width=2, color='white')
                    ),
                    name='Current Pattern',
                    hovertemplate=f'Current Pattern<br>Start: {pattern_dates[current_idx].strftime("%Y-%m-%d")}<extra></extra>'
                )
            )
        
        # Add cluster centers (k-means simplified)
        n_clusters = min(3, n_patterns)
        if n_clusters > 1:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(patterns_array)
            
            # Add cluster annotations
            for cluster_id in range(n_clusters):
                cluster_mask = clusters == cluster_id
                if cluster_mask.any():
                    cluster_center = patterns_2d[cluster_mask].mean(axis=0)
                    
                    fig.add_annotation(
                        x=cluster_center[0],
                        y=cluster_center[1],
                        text=f"Cluster {cluster_id+1}",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=self.color_palette['text'],
                        font=dict(size=12, color=self.color_palette['text'])
                    )
        
        fig.update_layout(
            title=f'{symbol} Pattern Space (PCA Reduced)',
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)",
            height=500,
            width=700,
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    def _calculate_similarity_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate similarity-based metrics."""
        metrics = {}
        
        if len(df) < 40:
            return metrics
        
        # Find top analogs
        analogs = self._find_top_historical_analogs(df, lookback=20, n_analogs=5)
        
        if not analogs:
            return metrics
        
        # Similarity statistics
        similarities = [sim for sim, _ in analogs]
        metrics['avg_top_analog_similarity'] = float(np.mean(similarities))
        metrics['max_analog_similarity'] = float(np.max(similarities))
        metrics['min_analog_similarity'] = float(np.min(similarities))
        metrics['analog_similarity_std'] = float(np.std(similarities))
        
        # Forward return statistics from analogs
        forward_returns_all = []
        for similarity, dates in analogs:
            forward_returns = self._get_forward_returns(df, dates[-1], window=10)
            if len(forward_returns) > 0:
                forward_returns_all.append(forward_returns.mean() * 100)
        
        if forward_returns_all:
            metrics['avg_forward_return_analogs'] = float(np.mean(forward_returns_all))
            metrics['std_forward_return_analogs'] = float(np.std(forward_returns_all))
            metrics['pct_positive_forward_returns'] = float(
                sum(1 for r in forward_returns_all if r > 0) / len(forward_returns_all) * 100
            )
        
        # Pattern consistency metrics
        lookback = 20
        n_comparisons = min(10, len(df) // lookback - 1)
        
        if n_comparisons > 1:
            pattern_similarities = []
            
            for i in range(n_comparisons):
                for j in range(i + 1, n_comparisons):
                    start_i = i * lookback
                    start_j = j * lookback
                    
                    pattern_i = df['close'].iloc[start_i:start_i + lookback]
                    pattern_j = df['close'].iloc[start_j:start_j + lookback]
                    
                    norm_i = self._normalize_series(pattern_i).values
                    norm_j = self._normalize_series(pattern_j).values
                    
                    distance = np.sqrt(np.mean((norm_i - norm_j) ** 2))
                    similarity = 1 - distance
                    pattern_similarities.append(similarity)
            
            if pattern_similarities:
                metrics['avg_pattern_similarity'] = float(np.mean(pattern_similarities))
                metrics['pattern_similarity_std'] = float(np.std(pattern_similarities))
        
        # Current pattern uniqueness
        current_similarity = similarities[0]  # Top analog similarity
        if 'avg_pattern_similarity' in metrics:
            metrics['pattern_uniqueness'] = float(
                (current_similarity - metrics['avg_pattern_similarity']) / 
                (metrics['pattern_similarity_std'] + 1e-8)
            )
        
        return metrics
    
    def _generate_similarity_insights(self, df: pd.DataFrame, metrics: Dict[str, float]) -> List[str]:
        """Generate textual insights from similarity analysis."""
        insights = []
        
        if not metrics:
            return insights
        
        # Pattern similarity insight
        max_sim = metrics.get('max_analog_similarity', 0)
        avg_sim = metrics.get('avg_top_analog_similarity', 0)
        
        if max_sim > 0.8:
            insights.append(f"Strong historical analog found (similarity: {max_sim:.2f}).")
        elif max_sim < 0.4:
            insights.append(f"Current pattern is relatively unique (best analog similarity: {max_sim:.2f}).")
        
        # Forward return insight from analogs
        avg_forward_return = metrics.get('avg_forward_return_analogs')
        if avg_forward_return is not None:
            if avg_forward_return > 2:
                insights.append(f"Historical analogs suggest positive forward returns (avg: {avg_forward_return:.1f}%).")
            elif avg_forward_return < -2:
                insights.append(f"Historical analogs suggest negative forward returns (avg: {avg_forward_return:.1f}%).")
        
        # Pattern consistency insight
        pattern_uniqueness = metrics.get('pattern_uniqueness')
        if pattern_uniqueness is not None:
            if abs(pattern_uniqueness) > 1:
                if pattern_uniqueness > 0:
                    insights.append(f"Current pattern is more typical than average (uniqueness z-score: {pattern_uniqueness:.1f}).")
                else:
                    insights.append(f"Current pattern is less typical than average (uniqueness z-score: {pattern_uniqueness:.1f}).")
        
        # Analog consistency insight
        sim_std = metrics.get('analog_similarity_std', 0)
        if sim_std < 0.1:
            insights.append(f"Multiple similar historical patterns found (consistent analogs).")
        elif sim_std > 0.2:
            insights.append(f"Wide range of historical analogs (diverse past patterns).")
        
        return insights
    
    def _find_historical_analogs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Find and analyze historical analogs in detail."""
        results = {}
        
        if len(df) < 40:
            return results
        
        # Find top analogs
        analogs = self._find_top_historical_analogs(df, lookback=20, n_analogs=10)
        
        if not analogs:
            return results
        
        results['top_analogs'] = []
        
        for i, (similarity, dates) in enumerate(analogs[:5]):
            analog_info = {
                'rank': i + 1,
                'similarity': float(similarity),
                'start_date': dates[0].strftime('%Y-%m-%d'),
                'end_date': dates[-1].strftime('%Y-%m-%d'),
                'duration_days': len(dates),
                'forward_returns': {}
            }
            
            # Calculate forward returns at various horizons
            for horizon in [5, 10, 20, 30]:
                forward_returns = self._get_forward_returns(df, dates[-1], window=horizon)
                if len(forward_returns) > 0:
                    analog_info['forward_returns'][f'{horizon}d'] = {
                        'mean': float(forward_returns.mean() * 100),
                        'std': float(forward_returns.std() * 100),
                        'max': float(forward_returns.max() * 100),
                        'min': float(forward_returns.min() * 100)
                    }
            
            results['top_analogs'].append(analog_info)
        
        # Aggregate statistics across analogs
        if results['top_analogs']:
            forward_returns_aggregated = {}
            
            for horizon in [5, 10, 20, 30]:
                horizon_returns = []
                for analog in results['top_analogs']:
                    if f'{horizon}d' in analog['forward_returns']:
                        horizon_returns.append(analog['forward_returns'][f'{horizon}d']['mean'])
                
                if horizon_returns:
                    forward_returns_aggregated[f'{horizon}d'] = {
                        'mean': float(np.mean(horizon_returns)),
                        'std': float(np.std(horizon_returns)),
                        'pct_positive': float(sum(1 for r in horizon_returns if r > 0) / len(horizon_returns) * 100)
                    }
            
            results['aggregated_forward_returns'] = forward_returns_aggregated
        
        # Pattern characteristics comparison
        current_pattern = df['close'].iloc[-20:]
        current_returns = current_pattern.pct_change().dropna()
        
        pattern_stats = {
            'current': {
                'return_mean': float(current_returns.mean() * 100),
                'return_std': float(current_returns.std() * 100),
                'total_return': float((current_pattern.iloc[-1] / current_pattern.iloc[0] - 1) * 100)
            }
        }
        
        for i, analog in enumerate(results['top_analogs'][:3]):
            analog_dates = analogs[i][1]
            analog_prices = df.loc[analog_dates, 'close']
            analog_returns = analog_prices.pct_change().dropna()
            
            pattern_stats[f'analog_{i+1}'] = {
                'return_mean': float(analog_returns.mean() * 100),
                'return_std': float(analog_returns.std() * 100),
                'total_return': float((analog_prices.iloc[-1] / analog_prices.iloc[0] - 1) * 100)
            }
        
        results['pattern_statistics'] = pattern_stats
        
        return results
    
    def _analyze_recurring_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze recurring patterns and their implications."""
        results = {}
        
        if len(df) < 100:
            return results
        
        lookback = 20
        n_patterns = min(50, len(df) // lookback)
        
        # Extract patterns
        patterns = []
        pattern_starts = []
        
        for i in range(0, n_patterns * lookback, lookback):
            if i + lookback <= len(df):
                pattern = df['close'].iloc[i:i + lookback]
                patterns.append(self._normalize_series(pattern).values)
                pattern_starts.append(df.index[i])
        
        if len(patterns) < 10:
            return results
        
        # Cluster patterns using simple distance matrix
        n_patterns = len(patterns)
        distance_matrix = np.zeros((n_patterns, n_patterns))
        
        for i in range(n_patterns):
            for j in range(i, n_patterns):
                if i == j:
                    distance = 0
                else:
                    distance = np.sqrt(np.mean((patterns[i] - patterns[j]) ** 2))
                
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        
        # Identify clusters (simplified threshold-based)
        threshold = 0.3
        clusters = []
        unassigned = list(range(n_patterns))
        
        while unassigned:
            # Start new cluster with first unassigned pattern
            cluster_start = unassigned.pop(0)
            cluster = [cluster_start]
            
            # Find similar patterns
            to_check = [cluster_start]
            
            while to_check:
                current = to_check.pop()
                
                # Find unassigned patterns similar to current
                for j in unassigned[:]:  # Copy list for safe removal
                    if distance_matrix[current, j] < threshold:
                        cluster.append(j)
                        unassigned.remove(j)
                        to_check.append(j)
            
            if len(cluster) > 1:  # Only keep clusters with multiple patterns
                clusters.append(cluster)
        
        results['pattern_clusters'] = []
        
        for i, cluster in enumerate(clusters[:5]):  # Limit to top 5 clusters
            cluster_info = {
                'cluster_id': i + 1,
                'size': len(cluster),
                'pattern_dates': [pattern_starts[idx].strftime('%Y-%m-%d') for idx in cluster],
                'average_similarity': float(
                    np.mean([distance_matrix[a, b] 
                            for a in cluster for b in cluster if a != b])
                )
            }
            
            # Calculate forward returns for patterns in this cluster
            forward_returns_by_horizon = {}
            
            for horizon in [5, 10, 20]:
                horizon_returns = []
                
                for pattern_idx in cluster:
                    start_date = pattern_starts[pattern_idx]
                    forward_returns = self._get_forward_returns(df, start_date + pd.Timedelta(days=lookback-1), 
                                                               window=horizon)
                    if len(forward_returns) > 0:
                        horizon_returns.append(forward_returns.mean() * 100)
                
                if horizon_returns:
                    forward_returns_by_horizon[f'{horizon}d'] = {
                        'mean': float(np.mean(horizon_returns)),
                        'std': float(np.std(horizon_returns)),
                        'pct_positive': float(
                            sum(1 for r in horizon_returns if r > 0) / len(horizon_returns) * 100
                        )
                    }
            
            cluster_info['forward_returns'] = forward_returns_by_horizon
            results['pattern_clusters'].append(cluster_info)
        
        # Check if current pattern belongs to any cluster
        current_pattern_idx = len(patterns) - 1  # Most recent pattern
        current_cluster_id = None
        
        for i, cluster in enumerate(clusters):
            if current_pattern_idx in cluster:
                current_cluster_id = i + 1
                break
        
        if current_cluster_id is not None:
            results['current_pattern_cluster'] = current_cluster_id
            current_cluster = clusters[current_cluster_id - 1]
            
            # Calculate similarity to cluster centroid
            cluster_patterns = [patterns[idx] for idx in current_cluster]
            centroid = np.mean(cluster_patterns, axis=0)
            current_to_centroid_distance = np.sqrt(
                np.mean((patterns[current_pattern_idx] - centroid) ** 2)
            )
            
            results['current_to_cluster_centroid_similarity'] = float(1 - current_to_centroid_distance)
            
            # Insight about cluster performance
            cluster_info = results['pattern_clusters'][current_cluster_id - 1]
            if 'forward_returns' in cluster_info and '10d' in cluster_info['forward_returns']:
                avg_return = cluster_info['forward_returns']['10d']['mean']
                if avg_return > 2:
                    results['cluster_performance_insight'] = f"Patterns in this cluster historically followed by positive returns (+{avg_return:.1f}% avg)"
                elif avg_return < -2:
                    results['cluster_performance_insight'] = f"Patterns in this cluster historically followed by negative returns ({avg_return:.1f}% avg)"
        
        return results
    
    def _analyze_cross_symbol_similarity(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze pattern similarity between different symbols."""
        results = {}
        
        symbols = list(data.keys())
        if len(symbols) < 2:
            return results
        
        # Extract recent patterns from each symbol
        lookback = 20
        patterns = {}
        
        for symbol, df in data.items():
            if len(df) >= lookback:
                recent_prices = df['close'].iloc[-lookback:]
                patterns[symbol] = self._normalize_series(recent_prices).values
        
        if len(patterns) < 2:
            return results
        
        # Calculate pairwise pattern similarity
        similarity_matrix = np.ones((len(symbols), len(symbols)))
        
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i != j and sym1 in patterns and sym2 in patterns:
                    pattern1 = patterns[sym1]
                    pattern2 = patterns[sym2]
                    
                    distance = np.sqrt(np.mean((pattern1 - pattern2) ** 2))
                    similarity = 1 - distance
                    similarity_matrix[i, j] = similarity
        
        # Create DataFrame for easier manipulation
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=symbols,
            columns=symbols
        )
        
        results['cross_symbol_pattern_similarity'] = similarity_df.to_dict()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=symbols,
            y=symbols,
            colorscale='RdBu',
            zmid=0.5,
            text=np.round(similarity_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverongaps=False,
            hovertemplate='Symbol 1: %{y}<br>Symbol 2: %{x}<br>Similarity: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Cross-Symbol Pattern Similarity (Recent 20-day patterns)',
            height=500,
            width=600,
            template='plotly_white'
        )
        
        results['cross_symbol_similarity_chart'] = fig
        
        # Generate insights
        insights = []
        
        # Find most similar pair
        most_similar = -1
        most_similar_pair = None
        
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i < j:  # Avoid duplicate comparisons
                    similarity = similarity_matrix[i, j]
                    if similarity > most_similar:
                        most_similar = similarity
                        most_similar_pair = (sym1, sym2)
        
        if most_similar_pair:
            insights.append(
                f"Most similar recent patterns: {most_similar_pair[0]}-{most_similar_pair[1]} "
                f"(similarity: {most_similar:.2f})"
            )
            
            if most_similar > 0.7:
                insights.append(f"Strong pattern alignment between {most_similar_pair[0]} and {most_similar_pair[1]}.")
        
        # Find least similar pair
        least_similar = 2
        least_similar_pair = None
        
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i < j:
                    similarity = similarity_matrix[i, j]
                    if similarity < least_similar:
                        least_similar = similarity
                        least_similar_pair = (sym1, sym2)
        
        if least_similar_pair and least_similar < 0.3:
            insights.append(
                f"Divergent patterns: {least_similar_pair[0]} and {least_similar_pair[1]} "
                f"show different recent behavior (similarity: {least_similar:.2f})"
            )
        
        results['insights'] = insights
        
        return results
    
    def _save_symbol_results(self, symbol: str, results: Dict[str, Any]) -> None:
        """
        FIXED: Merges similarity analysis results into the shared pkl file.
        Prevents overwriting data from other analysis modules.
        """
        try:
            # Ensure the directory exists
            self.artifacts_path.mkdir(parents=True, exist_ok=True)
            
            # Use the SHARED results file instead of a module-specific one
            output_file = self.artifacts_path / f"{symbol}_analysis_results.pkl"
            
            # 1. Prepare data for storage: remove charts (Plotly objects) to keep files small
            save_results = {k: v for k, v in results.items() if k not in ['charts', 'fig', 'plots']}
            
            # 2. Load existing results if they exist to perform a merge
            final_data = {}
            if output_file.exists():
                try:
                    final_data = joblib.load(output_file)
                    if not isinstance(final_data, dict):
                        final_data = {}
                except Exception:
                    final_data = {}
            
            # 3. Add/Update ONLY the similarity analysis key
            final_data['similarity'] = save_results
            
            # 4. Save the merged data back to the disk
            joblib.dump(final_data, output_file)
            self.logger.info(f"Merged similarity analysis for {symbol} into {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error merging similarity analysis for {symbol}: {e}")