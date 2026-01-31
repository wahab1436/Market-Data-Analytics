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
            # Use fixed seed for reproducibility across plots
            if len(X_test) > self.sample_size:
                np.random.seed(42)
                sample_indices = np.random.choice(len(X_test), self.sample_size, replace=False)
                X_sample = X_test[sample_indices]
            else:
                X_sample = X_test
                sample_indices = np.arange(len(X_test))
            
            # Compute SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Handle different SHAP value formats (Regression vs Classification)
            if isinstance(shap_values, list):
                # For binary/multiclass, take the class of interest (usually positive class 1)
                shap_values = shap_values[-1]
            
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
            # Use Try/Except as interaction values can be computationally expensive
            try:
                interaction_results = self._analyze_interactions(
                    explainer, X_sample, shap_values, feature_names, symbol
                )
                results['interactions'] = interaction_results
            except Exception as e:
                self.logger.warning(f"Could not compute interactions: {e}")
            
            # Generate insights
            insights = self._generate_shap_insights(summary_data, dependency_results, symbol)
            results['insights'] = insights
            
            # Save SHAP values (using proper merge logic)
            self._save_shap_values(shap_values, X_sample, feature_names, sample_indices, symbol)
            
            self.logger.info(f"SHAP computation completed for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error computing SHAP values: {e}")
            results['error'] = str(e)
        
        return results
    
    def _create_shap_summary(self, shap_values: np.ndarray, X: np.ndarray,
                            feature_names: List[str], symbol: str) -> Dict[str, Any]:
        """Create SHAP summary plot data."""
        summary = {}
        
        if len(shap_values.shape) != 2 or shap_values.shape[1] != len(feature_names):
            self.logger.warning(f"SHAP values shape {shap_values.shape} doesn't match feature names length {len(feature_names)}")
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
        
        colors = []
        for direction in sorted_direction:
            if direction > 0:
                colors.append(self.color_palette.get('danger', 'red'))
            else:
                colors.append(self.color_palette.get('success', 'green'))
        
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
        
        # Create beeswarm plot data
        beeswarm_data = self._create_beeswarm_data(shap_values, X, feature_names, sorted_indices)
        summary['beeswarm'] = beeswarm_data
        
        return summary
    
    def _create_beeswarm_data(self, shap_values: np.ndarray, X: np.ndarray,
                             feature_names: List[str], sorted_indices: np.ndarray) -> Dict[str, Any]:
        """Create data for beeswarm plot visualization."""
        beeswarm = {}
        
        top_indices = sorted_indices[:self.max_display]
        feature_data = []
        
        for i, feature_idx in enumerate(top_indices):
            feature_name = feature_names[feature_idx]
            feature_values = X[:, feature_idx]
            shap_for_feature = shap_values[:, feature_idx]
            
            # Sub-sample if data is large to keep charts responsive
            if len(feature_values) > 200:
                idx = np.random.choice(len(feature_values), 200, replace=False)
                feature_values = feature_values[idx]
                shap_for_feature = shap_for_feature[idx]
            
            # Calculate correlation safely
            corr = 0
            if len(feature_values) > 1 and np.std(feature_values) > 0 and np.std(shap_for_feature) > 0:
                 corr = float(np.corrcoef(feature_values, shap_for_feature)[0, 1])

            feature_data.append({
                'feature': feature_name,
                'values': feature_values.tolist(),
                'shap_values': shap_for_feature.tolist(),
                'correlation': corr
            })
        
        beeswarm['features'] = feature_data
        
        fig = go.Figure()
        
        for i, data in enumerate(feature_data[:10]):  # Limit to 10 features for clarity
            fig.add_trace(
                go.Scatter(
                    x=data['shap_values'],
                    y=[data['feature']] * len(data['shap_values']),
                    mode='markers',
                    name=data['feature'],
                    marker=dict(
                        size=6,
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
        
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:5]  # Top 5 features
        
        dependency_plots = []
        
        for feature_idx in top_indices:
            feature_name = feature_names[feature_idx]
            feature_values = X[:, feature_idx]
            shap_for_feature = shap_values[:, feature_idx]
            
            # Sort for smooth plotting
            sort_idx = np.argsort(feature_values)
            sorted_values = feature_values[sort_idx]
            sorted_shap = shap_for_feature[sort_idx]
            
            # Simple moving average smoothing
            window = max(3, len(sorted_values) // 20)
            if window > 1 and len(sorted_values) > window:
                smoothed_shap = np.convolve(sorted_shap, np.ones(window)/window, mode='valid')
                # Adjust x values to match convolution output size
                start = (window - 1) // 2
                end = start + len(smoothed_shap)
                smoothed_values = sorted_values[start:end]
            else:
                smoothed_shap = sorted_shap
                smoothed_values = sorted_values
            
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=smoothed_values,
                    y=smoothed_shap,
                    mode='lines+markers',
                    name='SHAP Dependency',
                    line=dict(color=self.color_palette.get('primary', 'blue'), width=2),
                    marker=dict(size=4),
                    hovertemplate='Feature Value: %{x:.2f}<br>SHAP Value: %{y:.4f}<extra></extra>'
                )
            )
            
            fig.add_hline(y=0, line_width=1, line_color='gray')
            
            fig.update_layout(
                title=f'{symbol}: SHAP Dependency - {feature_name}',
                height=400,
                showlegend=False,
                template='plotly_white',
                xaxis_title=f'{feature_name} (feature value)',
                yaxis_title='SHAP Value'
            )
            
            # Calculate stats
            corr = 0
            if len(feature_values) > 1 and np.std(feature_values) > 0:
                 corr = np.corrcoef(feature_values, shap_for_feature)[0, 1]

            dependency_plots.append({
                'feature': feature_name,
                'chart': fig,
                'statistics': {
                    'correlation': float(corr),
                    'mean_shap': float(np.mean(shap_for_feature)),
                    'std_shap': float(np.std(shap_for_feature))
                }
            })
        
        dependencies['plots'] = dependency_plots
        return dependencies
    
    def _create_force_plots(self, explainer: shap.TreeExplainer, X: np.ndarray,
                           shap_values: np.ndarray, feature_names: List[str],
                           symbol: str, sample_indices: np.ndarray) -> Dict[str, Any]:
        """Create force plot visualizations."""
        force_plots = {}
        if len(X) == 0:
            return force_plots
        
        n_samples = min(5, len(X))
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]
        
        predictions = base_value + np.sum(shap_values, axis=1)
        
        # Pick extreme and median samples
        highest_idx = np.argmax(predictions)
        lowest_idx = np.argmin(predictions)
        median_idx = np.argsort(predictions)[len(predictions) // 2]
        
        selected_indices = list(set([highest_idx, lowest_idx, median_idx]))[:n_samples]
        
        force_data = []
        for idx in selected_indices:
            sample_shap = shap_values[idx]
            sample_x = X[idx]
            prediction = predictions[idx]
            
            contributing_features = []
            for j in np.argsort(np.abs(sample_shap))[::-1][:10]:
                contributing_features.append({
                    'feature': feature_names[j],
                    'value': float(sample_x[j]),
                    'shap_value': float(sample_shap[j]),
                    'impact': 'increases' if sample_shap[j] > 0 else 'decreases'
                })
            
            force_data.append({
                'sample_index': int(sample_indices[idx]),
                'prediction': float(prediction),
                'base_value': float(base_value),
                'contributing_features': contributing_features
            })
        
        force_plots['samples'] = force_data
        
        # Median prediction chart
        if force_data:
            median_sample = force_data[-1] 
            fig = go.Figure()
            
            feats = [f['feature'] for f in median_sample['contributing_features'][:8]]
            vals = [f['shap_value'] for f in median_sample['contributing_features'][:8]]
            cols = [self.color_palette.get('danger', 'red') if v > 0 
                   else self.color_palette.get('success', 'green') for v in vals]
            
            fig.add_trace(go.Bar(
                x=[median_sample['base_value']], y=['Base Value'], orientation='h',
                marker_color='gray', name='Base Value'
            ))
            
            fig.add_trace(go.Bar(
                x=vals, y=feats, orientation='h', marker_color=cols, name='Features'
            ))
            
            fig.update_layout(
                title=f'{symbol}: Force Plot (Median Prediction)',
                height=400, showlegend=False, template='plotly_white', barmode='relative'
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
        
        avg_shap = np.mean(shap_values, axis=0)
        top_indices = np.argsort(np.abs(avg_shap))[::-1][:self.max_display]
        
        top_features = [feature_names[i] for i in top_indices]
        top_vals = [avg_shap[i] for i in top_indices]
        
        waterfall_data = []
        cumulative = float(base_value)
        waterfall_data.append({'feature': 'Base', 'value': cumulative, 'cumulative': cumulative})
        
        for f, v in zip(top_features, top_vals):
            cumulative += v
            waterfall_data.append({'feature': f, 'value': v, 'cumulative': cumulative})
            
        waterfall_data.append({'feature': 'Final', 'value': cumulative - base_value, 'cumulative': cumulative})
        waterfall['data'] = waterfall_data
        
        fig = go.Figure()
        
        feats = [d['feature'] for d in waterfall_data]
        vals = [d['value'] for d in waterfall_data]
        
        fig.add_trace(go.Waterfall(
            name="SHAP", orientation="v",
            measure=['absolute'] + ['relative']*(len(waterfall_data)-2) + ['total'],
            x=feats, y=vals,
            connector={"line": {"color": "gray"}},
            decreasing={"marker": {"color": self.color_palette.get('success', 'green')}},
            increasing={"marker": {"color": self.color_palette.get('danger', 'red')}},
            totals={"marker": {"color": self.color_palette.get('primary', 'blue')}}
        ))
        
        fig.update_layout(title=f'{symbol}: SHAP Waterfall (Avg)', height=600, template='plotly_white')
        waterfall['chart'] = fig
        return waterfall

    def _analyze_interactions(self, explainer: shap.TreeExplainer, X: np.ndarray,
                             shap_values: np.ndarray, feature_names: List[str],
                             symbol: str) -> Dict[str, Any]:
        """Analyze feature interactions."""
        interactions = {}
        
        # Use SHAP interaction values if available (computationally expensive)
        # Here we use a correlation proxy for speed as per original design
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:10]
        top_features = [feature_names[i] for i in top_indices]
        
        matrix = np.zeros((len(top_indices), len(top_indices)))
        
        for i, idx_i in enumerate(top_indices):
            for j, idx_j in enumerate(top_indices):
                if i <= j:
                    corr = np.corrcoef(shap_values[:, idx_i], shap_values[:, idx_j])[0, 1]
                    matrix[i, j] = matrix[j, i] = corr
                    
        interactions['matrix'] = {'features': top_features, 'values': matrix.tolist()}
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix, x=top_features, y=top_features, colorscale='RdBu', zmid=0
        ))
        fig.update_layout(title=f'{symbol}: SHAP Interaction Matrix', height=600)
        interactions['chart'] = fig
        
        return interactions

    def _generate_shap_insights(self, summary: Dict[str, Any], 
                               dependencies: Dict[str, Any], symbol: str) -> List[str]:
        """Generate text insights."""
        insights = []
        if 'feature_importance' in summary:
            top = summary['feature_importance'][:3]
            txt = [f"{f['feature']} ({'increases' if f['direction']>0 else 'decreases'})" for f in top]
            insights.append(f"Top drivers: {', '.join(txt)}")
        return insights

    def _save_shap_values(self, shap_values: np.ndarray, X: np.ndarray,
                         feature_names: List[str], sample_indices: np.ndarray,
                         symbol: str):
        """
        FIXED: Saves SHAP data to the unified _analysis_results.pkl file.
        Uses Load-Update-Save pattern to preserve data from other modules.
        """
        try:
            self.artifacts_path.mkdir(parents=True, exist_ok=True)
            output_file = self.artifacts_path / "_analysis_results.pkl"
            
            # Load existing
            if output_file.exists():
                try:
                    combined_data = joblib.load(output_file)
                except Exception as e:
                    self.logger.warning(f"Could not load existing results: {e}")
                    combined_data = {}
            else:
                combined_data = {}
            
            # Ensure 'explainability' key exists in unified structure
            if 'explainability' not in combined_data:
                combined_data['explainability'] = {}
            
            # Update this symbol's SHAP data (nested under explainability key)
            combined_data['explainability'][symbol] = {
                'shap_values': shap_values,
                'X_sample': X,
                'feature_names': feature_names,
                'sample_indices': sample_indices,
                'timestamp': pd.Timestamp.now()
            }
            
            # Save back
            joblib.dump(combined_data, output_file)
            self.logger.info(f"Merged SHAP results for {symbol} into {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving SHAP values: {e}")

    def create_dashboard_visualizations(self, symbol: str) -> Dict[str, Any]:
        """Generate charts for dashboard from saved artifacts."""
        vis = {}
        output_file = self.artifacts_path / "_analysis_results.pkl"
        
        if not output_file.exists():
            return vis
            
        try:
            data = joblib.load(output_file)
            
            # Access explainability data for this symbol
            if 'explainability' not in data or symbol not in data['explainability']:
                self.logger.warning(f"No SHAP data found for {symbol}")
                return vis
            
            shap_data = data['explainability'][symbol]
            
            if shap_data:
                # Re-generate key charts on demand
                summary = self._create_shap_summary(
                    shap_data['shap_values'], shap_data['X_sample'], 
                    shap_data['feature_names'], symbol
                )
                if 'chart' in summary:
                    vis['summary_chart'] = summary['chart']
                    
                if 'beeswarm' in summary and 'chart' in summary['beeswarm']:
                    vis['beeswarm_chart'] = summary['beeswarm']['chart']
                    
        except Exception as e:
            self.logger.error(f"Error creating dashboard vis: {e}")
            
        return vis