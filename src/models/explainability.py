"""
Explainability Module
SHAP analysis for XGBoost model interpretability
Batch-computed only, never computed in dashboard
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ModelExplainability:
    """SHAP-based explainability for XGBoost models (batch-only computation)."""
    
    def __init__(self, config: Dict[str, Any], logger):
        """Initialize explainability analyzer."""
        self.config = config
        self.logger = logger
        self.color_palette = config['dashboard']['color_palette']
        self.models_path = Path(config['paths']['models'])
        self.artifacts_path = Path(config['paths']['artifacts'])
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # SHAP parameters
        self.sample_size = 100  # Sample size for SHAP computation
        self.max_display = 15   # Maximum features to display
    
    def compute_shap(self, model: Any, X_test: np.ndarray, 
                    feature_names: List[str], symbol: str = None) -> Dict[str, Any]:
        """Compute SHAP values for XGBoost model (batch computation only)."""
        self.logger.info(f"Computing SHAP explainability for {symbol if symbol else 'model'}")
        
        results = {}
        
        if model is None or X_test is None or len(X_test) == 0:
            self.logger.warning("No model or test data for SHAP computation")
            return results
        
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            
            # Sample data for SHAP computation (for performance)
            if len(X_test) > self.sample_size:
                sample_indices = np.random.choice(len(X_test), self.sample_size, replace=False)
                X_sample = X_test[sample_indices]
            else:
                X_sample = X_test
                sample_indices = np.arange(len(X_test))
            
            # Compute SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Take first for regression
            
            # Ensure shap_values is 2D
            if len(shap_values.shape) == 1:
                shap_values = shap_values.reshape(-1, 1)
            
            # Create summary plots
            summary_data = self._create_shap_summary(shap_values, X_sample, feature_names, symbol)
            results['summary'] = summary_data
            
            # Create dependency plots for top features
            dependency_results = self._create_dependency_plots(
                explainer, X_sample, shap_values, feature_names, symbol
            )
            results['dependencies'] = dependency_results
            
            # Create force plots for specific predictions
            force_results = self._create_force_plots(
                explainer, X_sample, shap_values, feature_names, symbol, sample_indices
            )
            results['force_plots'] = force_results
            
            # Create waterfall plot for average prediction
            waterfall_data = self._create_waterfall_plot(
                explainer, X_sample, shap_values, feature_names, symbol
            )
            results['waterfall'] = waterfall_data
            
            # Create interaction analysis
            interaction_results = self._analyze_interactions(
                explainer, X_sample, shap_values, feature_names, symbol
            )
            results['interactions'] = interaction_results
            
            # Generate insights
            insights = self._generate_shap_insights(summary_data, dependency_results, symbol)
            results['insights'] = insights
            
            # Save SHAP values for dashboard
            self._save_shap_values(shap_values, X_sample, feature_names, sample_indices, symbol)
            
            self.logger.info(f"SHAP computation completed for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error computing SHAP values: {e}")
            # Return empty results rather than failing
            results['error'] = str(e)
        
        return results
    
    def _create_shap_summary(self, shap_values: np.ndarray, X: np.ndarray,
                            feature_names: List[str], symbol: str) -> Dict[str, Any]:
        """Create SHAP summary plot data."""
        summary = {}
        
        if len(shap_values.shape) != 2 or shap_values.shape[1] != len(feature_names):
            self.logger.warning("SHAP values shape doesn't match feature names")
            return summary
        
        # Calculate mean absolute SHAP values (feature importance)
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Sort features by importance
        sorted_indices = np.argsort(mean_abs_shap)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices[:self.max_display]]
        sorted_importance = [mean_abs_shap[i] for i in sorted_indices[:self.max_display]]
        
        # Get direction of impact (average SHAP value)
        mean_shap = np.mean(shap_values, axis=0)
        sorted_direction = [mean_shap[i] for i in sorted_indices[:self.max_display]]
        
        # Create summary DataFrame
        summary_df = pd.DataFrame({
            'feature': sorted_features,
            'importance': sorted_importance,
            'direction': sorted_direction,
            'abs_importance': np.abs(sorted_direction)
        })
        
        summary['feature_importance'] = summary_df.to_dict('records')
        
        # Create summary plot
        fig = go.Figure()
        
        # Color by direction
        colors = []
        for direction in sorted_direction:
            if direction > 0:
                colors.append(self.color_palette['danger'])  # Red for positive impact
            else:
                colors.append(self.color_palette['success'])  # Green for negative impact
        
        fig.add_trace(
            go.Bar(
                x=sorted_importance,
                y=sorted_features,
                orientation='h',
                marker_color=colors,
                text=[f'{imp:.4f}' for imp in sorted_importance],
                textposition='auto',
                hovertemplate='Feature: %{y}<br>Importance: %{x:.4f}<br>Direction: %{customdata:.4f}',
                customdata=sorted_direction
            )
        )
        
        fig.update_layout(
            title=f'{symbol}: SHAP Feature Importance',
            height=500,
            showlegend=False,
            template='plotly_white',
            xaxis_title='Mean |SHAP Value| (average impact on model output)',
            yaxis_title='Feature',
            yaxis=dict(categoryorder='total ascending')
        )
        
        summary['chart'] = fig
        
        # Create beeswarm plot data (simplified)
        beeswarm_data = self._create_beeswarm_data(shap_values, X, feature_names, sorted_indices)
        summary['beeswarm'] = beeswarm_data
        
        return summary
    
    def _create_beeswarm_data(self, shap_values: np.ndarray, X: np.ndarray,
                             feature_names: List[str], sorted_indices: np.ndarray) -> Dict[str, Any]:
        """Create data for beeswarm plot visualization."""
        beeswarm = {}
        
        # Limit to top features
        top_indices = sorted_indices[:self.max_display]
        top_features = [feature_names[i] for i in top_indices]
        
        # Prepare data for each feature
        feature_data = []
        
        for i, feature_idx in enumerate(top_indices):
            feature_name = feature_names[feature_idx]
            feature_values = X[:, feature_idx]
            shap_for_feature = shap_values[:, feature_idx]
            
            # Sample for visualization (limit points)
            if len(feature_values) > 100:
                sample_idx = np.random.choice(len(feature_values), 100, replace=False)
                feature_values = feature_values[sample_idx]
                shap_for_feature = shap_for_feature[sample_idx]
            
            feature_data.append({
                'feature': feature_name,
                'values': feature_values.tolist(),
                'shap_values': shap_for_feature.tolist(),
                'correlation': float(np.corrcoef(feature_values, shap_for_feature)[0, 1]) 
                             if len(feature_values) > 1 else 0
            })
        
        beeswarm['features'] = feature_data
        
        # Create simplified beeswarm chart
        fig = go.Figure()
        
        for i, data in enumerate(feature_data[:10]):  # Limit to 10 features for clarity
            fig.add_trace(
                go.Scatter(
                    x=data['shap_values'],
                    y=[data['feature']] * len(data['shap_values']),
                    mode='markers',
                    name=data['feature'],
                    marker=dict(
                        size=8,
                        color=data['values'],
                        colorscale='RdBu',
                        showscale=(i == 0),
                        colorbar=dict(title="Feature Value" if i == 0 else None),
                        opacity=0.6
                    ),
                    hovertemplate='SHAP: %{x:.4f}<br>Value: %{marker.color:.2f}<extra></extra>'
                )
            )
        
        fig.update_layout(
            title='SHAP Beeswarm Plot (Top 10 Features)',
            height=600,
            showlegend=False,
            template='plotly_white',
            xaxis_title='SHAP Value (impact on model output)',
            yaxis_title='Feature'
        )
        
        beeswarm['chart'] = fig
        
        return beeswarm
    
    def _create_dependency_plots(self, explainer: shap.TreeExplainer, X: np.ndarray,
                                shap_values: np.ndarray, feature_names: List[str],
                                symbol: str) -> Dict[str, Any]:
        """Create SHAP dependency plots for top features."""
        dependencies = {}
        
        # Get top features by importance
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:5]  # Top 5 features
        
        dependency_plots = []
        
        for feature_idx in top_indices:
            feature_name = feature_names[feature_idx]
            
            # Create dependency plot data
            feature_values = X[:, feature_idx]
            shap_for_feature = shap_values[:, feature_idx]
            
            # Sort by feature value for smooth plot
            sort_idx = np.argsort(feature_values)
            sorted_values = feature_values[sort_idx]
            sorted_shap = shap_for_feature[sort_idx]
            
            # Apply smoothing for better visualization
            window = max(3, len(sorted_values) // 20)
            if window > 1 and len(sorted_values) > window:
                smoothed_shap = np.convolve(sorted_shap, np.ones(window)/window, mode='valid')
                smoothed_values = sorted_values[window//2: -window//2 + 1]
            else:
                smoothed_shap = sorted_shap
                smoothed_values = sorted_values
            
            # Create plot
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=smoothed_values,
                    y=smoothed_shap,
                    mode='lines+markers',
                    name='SHAP Dependency',
                    line=dict(color=self.color_palette['primary'], width=2),
                    marker=dict(size=4),
                    hovertemplate='Feature Value: %{x:.2f}<br>SHAP Value: %{y:.4f}<extra></extra>'
                )
            )
            
            # Add horizontal line at 0
            fig.add_hline(
                y=0,
                line_width=1,
                line_color=self.color_palette['text']
            )
            
            fig.update_layout(
                title=f'{symbol}: SHAP Dependency - {feature_name}',
                height=400,
                showlegend=False,
                template='plotly_white',
                xaxis_title=f'{feature_name} (feature value)',
                yaxis_title='SHAP Value (impact on model output)'
            )
            
            # Calculate statistics
            corr = np.corrcoef(feature_values, shap_for_feature)[0, 1] if len(feature_values) > 1 else 0
            
            dependency_plots.append({
                'feature': feature_name,
                'chart': fig,
                'statistics': {
                    'correlation': float(corr),
                    'mean_shap': float(np.mean(shap_for_feature)),
                    'std_shap': float(np.std(shap_for_feature)),
                    'mean_feature_value': float(np.mean(feature_values)),
                    'std_feature_value': float(np.std(feature_values))
                }
            })
        
        dependencies['plots'] = dependency_plots
        
        return dependencies
    
    def _create_force_plots(self, explainer: shap.TreeExplainer, X: np.ndarray,
                           shap_values: np.ndarray, feature_names: List[str],
                           symbol: str, sample_indices: np.ndarray) -> Dict[str, Any]:
        """Create force plot visualizations for specific predictions."""
        force_plots = {}
        
        if len(X) == 0:
            return force_plots
        
        # Select a few interesting samples
        n_samples = min(5, len(X))
        
        # Find samples with highest and lowest predictions
        base_value = explainer.expected_value
        
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]
        
        predictions = base_value + np.sum(shap_values, axis=1)
        
        # Get extreme predictions
        highest_idx = np.argmax(predictions)
        lowest_idx = np.argmin(predictions)
        median_idx = np.argsort(predictions)[len(predictions) // 2]
        
        selected_indices = [highest_idx, lowest_idx, median_idx][:n_samples]
        
        force_data = []
        
        for i, idx in enumerate(selected_indices):
            sample_shap = shap_values[idx]
            sample_x = X[idx]
            prediction = predictions[idx]
            
            # Get top contributing features
            contributing_features = []
            for j in np.argsort(np.abs(sample_shap))[::-1][:10]:  # Top 10 contributors
                feature_value = sample_x[j]
                shap_value = sample_shap[j]
                
                contributing_features.append({
                    'feature': feature_names[j],
                    'value': float(feature_value),
                    'shap_value': float(shap_value),
                    'impact': 'increases' if shap_value > 0 else 'decreases'
                })
            
            force_data.append({
                'sample_index': int(sample_indices[idx]),
                'prediction': float(prediction),
                'base_value': float(base_value),
                'contributing_features': contributing_features,
                'total_shap': float(np.sum(sample_shap))
            })
        
        force_plots['samples'] = force_data
        
        # Create a visualization for the median prediction
        if force_data:
            median_sample = force_data[-1]  # Last one is median
            
            fig = go.Figure()
            
            # Prepare data for waterfall-style plot
            features = [f['feature'] for f in median_sample['contributing_features'][:8]]
            shap_contributions = [f['shap_value'] for f in median_sample['contributing_features'][:8]]
            colors = [self.color_palette['danger'] if c > 0 
                     else self.color_palette['success'] for c in shap_contributions]
            
            # Add base value bar
            fig.add_trace(go.Bar(
                x=[median_sample['base_value']],
                y=['Base Value'],
                orientation='h',
                marker_color=self.color_palette['text'],
                name='Base Value',
                hovertemplate='Base Value: %{x:.4f}<extra></extra>'
            ))
            
            # Add feature contributions
            fig.add_trace(go.Bar(
                x=shap_contributions,
                y=features,
                orientation='h',
                marker_color=colors,
                name='Feature Contributions',
                hovertemplate='Feature: %{y}<br>Contribution: %{x:.4f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'{symbol}: Force Plot - Median Prediction',
                height=400,
                showlegend=False,
                template='plotly_white',
                xaxis_title='Contribution to Prediction',
                yaxis_title='Feature',
                yaxis=dict(categoryorder='total ascending'),
                barmode='relative'
            )
            
            force_plots['median_prediction_chart'] = fig
        
        return force_plots
    
    def _create_waterfall_plot(self, explainer: shap.TreeExplainer, X: np.ndarray,
                              shap_values: np.ndarray, feature_names: List[str],
                              symbol: str) -> Dict[str, Any]:
        """Create waterfall plot for average prediction."""
        waterfall = {}
        
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]
        
        # Calculate average SHAP values
        avg_shap = np.mean(shap_values, axis=0)
        
        # Get top contributing features
        top_indices = np.argsort(np.abs(avg_shap))[::-1][:self.max_display]
        top_features = [feature_names[i] for i in top_indices]
        top_shap = [avg_shap[i] for i in top_indices]
        
        # Sort for waterfall
        sorted_indices = np.argsort(top_shap)[::-1]  # Sort by SHAP value
        sorted_features = [top_features[i] for i in sorted_indices]
        sorted_shap = [top_shap[i] for i in sorted_indices]
        
        # Create waterfall data
        waterfall_data = []
        cumulative = float(base_value)
        
        waterfall_data.append({
            'feature': 'Base Value',
            'value': cumulative,
            'cumulative': cumulative,
            'is_total': False
        })
        
        for feature, shap_value in zip(sorted_features, sorted_shap):
            cumulative += shap_value
            waterfall_data.append({
                'feature': feature,
                'value': shap_value,
                'cumulative': cumulative,
                'is_total': False
            })
        
        # Add final prediction
        waterfall_data.append({
            'feature': 'Final Prediction',
            'value': cumulative - float(base_value),
            'cumulative': cumulative,
            'is_total': True
        })
        
        waterfall['data'] = waterfall_data
        
        # Create waterfall chart
        fig = go.Figure()
        
        # Prepare data for plotting
        features = [d['feature'] for d in waterfall_data]
        values = [d['value'] for d in waterfall_data]
        cumulatives = [d['cumulative'] for d in waterfall_data]
        
        # Determine colors
        colors = []
        for d in waterfall_data:
            if d['feature'] == 'Base Value':
                colors.append(self.color_palette['text'])
            elif d['feature'] == 'Final Prediction':
                colors.append(self.color_palette['primary'])
            elif d['value'] > 0:
                colors.append(self.color_palette['danger'])
            else:
                colors.append(self.color_palette['success'])
        
        # Create measure for waterfall
        measure = ['absolute'] + ['relative'] * (len(waterfall_data) - 2) + ['total']
        
        fig.add_trace(go.Waterfall(
            name="SHAP",
            orientation="v",
            measure=measure,
            x=features,
            y=values,
            text=[f'{v:.4f}' for v in values],
            textposition="outside",
            connector={"line": {"color": self.color_palette['text']}},
            decreasing={"marker": {"color": self.color_palette['success']}},
            increasing={"marker": {"color": self.color_palette['danger']}},
            totals={"marker": {"color": self.color_palette['primary']}}
        ))
        
        # Add cumulative line
        fig.add_trace(go.Scatter(
            x=features,
            y=cumulatives,
            mode='lines+markers',
            name='Cumulative',
            line=dict(color=self.color_palette['warning'], width=2, dash='dot'),
            marker=dict(size=8),
            yaxis='y2',
            hovertemplate='Cumulative: %{y:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'{symbol}: SHAP Waterfall Plot (Average Prediction)',
            height=600,
            showlegend=True,
            template='plotly_white',
            xaxis_title='Feature',
            yaxis_title='SHAP Value Contribution',
            yaxis2=dict(
                title='Cumulative Value',
                overlaying='y',
                side='right'
            )
        )
        
        waterfall['chart'] = fig
        
        return waterfall
    
    def _analyze_interactions(self, explainer: shap.TreeExplainer, X: np.ndarray,
                             shap_values: np.ndarray, feature_names: List[str],
                             symbol: str) -> Dict[str, Any]:
        """Analyze feature interactions using SHAP."""
        interactions = {}
        
        # Get top features
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:10]
        top_features = [feature_names[i] for i in top_indices]
        
        # Calculate pairwise interactions (simplified)
        interaction_matrix = np.zeros((len(top_indices), len(top_indices)))
        
        for i, idx_i in enumerate(top_indices):
            for j, idx_j in enumerate(top_indices):
                if i < j:
                    # Simple correlation of SHAP values as proxy for interaction
                    corr = np.corrcoef(shap_values[:, idx_i], shap_values[:, idx_j])[0, 1]
                    interaction_matrix[i, j] = corr
                    interaction_matrix[j, i] = corr
                elif i == j:
                    interaction_matrix[i, j] = 1.0
        
        interactions['matrix'] = {
            'features': top_features,
            'values': interaction_matrix.tolist()
        }
        
        # Create interaction heatmap
        fig = go.Figure(data=go.Heatmap(
            z=interaction_matrix,
            x=top_features,
            y=top_features,
            colorscale='RdBu',
            zmid=0,
            text=np.round(interaction_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='Feature 1: %{y}<br>Feature 2: %{x}<br>Interaction: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'{symbol}: SHAP Interaction Matrix (Top 10 Features)',
            height=600,
            width=700,
            template='plotly_white'
        )
        
        interactions['chart'] = fig
        
        # Find strongest interactions
        strong_interactions = []
        for i in range(len(top_indices)):
            for j in range(i+1, len(top_indices)):
                strength = abs(interaction_matrix[i, j])
                if strength > 0.3:  # Threshold for strong interaction
                    strong_interactions.append({
                        'feature1': top_features[i],
                        'feature2': top_features[j],
                        'correlation': float(interaction_matrix[i, j]),
                        'strength': float(strength)
                    })
        
        # Sort by strength
        strong_interactions.sort(key=lambda x: x['strength'], reverse=True)
        interactions['strong_interactions'] = strong_interactions[:5]  # Top 5
        
        return interactions
    
    def _generate_shap_insights(self, summary: Dict[str, Any], 
                               dependencies: Dict[str, Any], symbol: str) -> List[str]:
        """Generate insights from SHAP analysis."""
        insights = []
        
        if not summary:
            return insights
        
        # Top feature insights
        if 'feature_importance' in summary:
            top_features = summary['feature_importance'][:3]
            
            feature_insights = []
            for feature in top_features:
                direction = "increases" if feature['direction'] > 0 else "decreases"
                feature_insights.append(
                    f"{feature['feature']} ({direction} volatility)"
                )
            
            insights.append(f"Most influential features: {', '.join(feature_insights)}.")
        
        # Model behavior insights
        if 'dependencies' in dependencies and 'plots' in dependencies:
            for plot_data in dependencies['plots'][:2]:  # First 2 features
                stats = plot_data['statistics']
                feature = plot_data['feature']
                corr = stats['correlation']
                
                if abs(corr) > 0.3:
                    relationship = "positive" if corr > 0 else "negative"
                    insights.append(
                        f"Strong {relationship} relationship between {feature} and its impact."
                    )
        
        # Model complexity insight
        insights.append("SHAP analysis reveals non-linear feature relationships captured by XGBoost.")
        
        # Model trust insight
        if 'feature_importance' in summary:
            top_importance = summary['feature_importance'][0]['importance']
            if top_importance > 0.1:
                insights.append("Model decisions are driven by a few key features, enhancing interpretability.")
            else:
                insights.append("Feature importance is distributed, suggesting complex interactions.")
        
        return insights
    
    def _save_shap_values(self, shap_values: np.ndarray, X: np.ndarray,
                         feature_names: List[str], sample_indices: np.ndarray,
                         symbol: str):
        """Save SHAP values and data for dashboard visualization."""
        shap_data = {
            'shap_values': shap_values,
            'X_sample': X,
            'feature_names': feature_names,
            'sample_indices': sample_indices,
            'timestamp': pd.Timestamp.now()
        }
        
        shap_path = self.artifacts_path / f"{symbol}_shap_values.pkl"
        joblib.dump(shap_data, shap_path, compress=3)
        
        self.logger.debug(f"Saved SHAP values for {symbol}")
    
    def load_shap_values(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Load precomputed SHAP values from disk."""
        shap_path = self.artifacts_path / f"{symbol}_shap_values.pkl"
        
        if not shap_path.exists():
            self.logger.warning(f"No SHAP values found for {symbol}")
            return None
        
        try:
            shap_data = joblib.load(shap_path)
            self.logger.info(f"Loaded SHAP values for {symbol}")
            return shap_data
        except Exception as e:
            self.logger.error(f"Error loading SHAP values for {symbol}: {e}")
            return None
    
    def create_dashboard_visualizations(self, symbol: str) -> Dict[str, Any]:
        """Create visualizations from precomputed SHAP values (dashboard-safe)."""
        visualizations = {}
        
        # Load precomputed SHAP values
        shap_data = self.load_shap_values(symbol)
        
        if shap_data is None:
            return visualizations
        
        # Recreate charts from saved data
        shap_values = shap_data['shap_values']
        X_sample = shap_data['X_sample']
        feature_names = shap_data['feature_names']
        
        # Create summary chart
        summary_data = self._create_shap_summary(shap_values, X_sample, feature_names, symbol)
        if 'chart' in summary_data:
            visualizations['summary_chart'] = summary_data['chart']
        
        # Create dependency charts for top 2 features
        if 'feature_importance' in summary_data:
            top_features = [item['feature'] for item in summary_data['feature_importance'][:2]]
            
            for feature_name in top_features:
                if feature_name in feature_names:
                    feature_idx = feature_names.index(feature_name)
                    
                    # Create simplified dependency plot
                    fig = go.Figure()
                    
                    feature_values = X_sample[:, feature_idx]
                    shap_for_feature = shap_values[:, feature_idx]
                    
                    # Scatter plot
                    fig.add_trace(
                        go.Scatter(
                            x=feature_values,
                            y=shap_for_feature,
                            mode='markers',
                            name=feature_name,
                            marker=dict(
                                size=8,
                                color=shap_for_feature,
                                colorscale='RdBu',
                                showscale=True,
                                colorbar=dict(title="SHAP Value")
                            ),
                            hovertemplate='Feature Value: %{x:.2f}<br>SHAP Value: %{y:.4f}<extra></extra>'
                        )
                    )
                    
                    fig.update_layout(
                        title=f'{symbol}: SHAP Dependency - {feature_name}',
                        height=400,
                        showlegend=False,
                        template='plotly_white',
                        xaxis_title=f'{feature_name} Value',
                        yaxis_title='SHAP Value'
                    )
                    
                    visualizations[f'dependency_{feature_name}'] = fig
        
        # Create beeswarm chart if available
        if 'beeswarm' in summary_data and 'chart' in summary_data['beeswarm']:
            visualizations['beeswarm_chart'] = summary_data['beeswarm']['chart']
        
        return visualizations
