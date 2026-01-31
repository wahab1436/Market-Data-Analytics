"""
KNN Similarity Module
K-Nearest Neighbors for historical pattern matching and analog discovery
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class KNNSimilarity:
    """KNN models for historical similarity analysis and pattern matching."""
    
    def __init__(self, config: Dict[str, Any], logger):
        """Initialize KNN modeler."""
        self.config = config
        self.logger = logger
        self.color_palette = config['dashboard']['color_palette']
        self.models_path = Path(config['paths']['models'])
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.artifacts_path = Path(config['paths']['artifacts'])
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # KNN parameters
        self.n_neighbors = config['models']['knn']['n_neighbors']
        self.metric = config['models']['knn']['metric']
        self.test_size = config['models']['validation']['test_size']
    
    def train_and_evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Train and evaluate KNN models for similarity and prediction."""
        results = {}
        
        by_symbol = data['by_symbol']
        
        for symbol, df in by_symbol.items():
            self.logger.info(f"Training KNN models for {symbol}")
            
            symbol_results = {
                'charts': {},
                'insights': [],
                'metrics': {},
                'models': {},
                'similarities': {},
                'analogs': {}
            }
            
            # Prepare data for KNN
            X, y, feature_names = self._prepare_knn_data(df, symbol)
            
            if X is None or len(X) < self.n_neighbors * 2:
                self.logger.warning(f"Insufficient data for KNN on {symbol}")
                continue
            
            # Split data chronologically
            X_train, X_test, y_train, y_test, train_dates, test_dates = self._chronological_split(
                X, y, df.index
            )
            
            # Train KNN models
            models = self._train_models(X_train, y_train, symbol)
            symbol_results['models'] = models
            
            # Evaluate models
            evaluation = self._evaluate_models(models, X_test, y_test, symbol)
            symbol_results['metrics'].update(evaluation)
            
            # Find historical analogs
            analogs = self._find_historical_analogs(models['similarity'], X, df.index, symbol)
            symbol_results['analogs'] = analogs
            
            # Calculate similarity matrix
            similarities = self._calculate_similarities(models['similarity'], X, symbol)
            symbol_results['similarities'] = similarities
            
            # Make predictions
            predictions = self._make_predictions(models['regressor'], X_test, y_test, test_dates, symbol)
            symbol_results['predictions'] = predictions
            
            # Create KNN charts
            knn_charts = self._create_knn_charts(
                models, X, X_test, y_test, train_dates, test_dates, 
                analogs, symbol, feature_names
            )
            symbol_results['charts'].update(knn_charts)
            
            # Generate insights
            insights = self._generate_knn_insights(models, evaluation, analogs, symbol)
            symbol_results['insights'].extend(insights)
            
            # Save models
            self._save_models(models, symbol)
            
            # Save results to artifacts folder
            self._save_symbol_results(symbol, symbol_results)
            
            results[symbol] = symbol_results
        
        # Cross-symbol KNN analysis
        if len(by_symbol) > 1:
            cross_results = self._analyze_cross_symbol_knn(results, by_symbol)
            results['cross_symbol'] = cross_results
        
        return results
    
    def _prepare_knn_data(self, df: pd.DataFrame, symbol: str) -> Tuple:
        """Prepare data for KNN models."""
        # Define features for similarity search
        feature_categories = [
            'return', 'log_return',
            'volume_ratio_10d', 'volume_ratio_20d',
            'volatility_10d', 'volatility_20d',
            'ma_20d', 'ma_50d',
            'price_vs_ma_20d_pct', 'price_vs_ma_50d_pct',
            'daily_range_pct', 'atr_14d',
            'return_lag_1d', 'return_lag_5d'
        ]
        
        # Select available features
        available_features = [f for f in feature_categories if f in df.columns]
        
        if len(available_features) < 5:
            self.logger.warning(f"Insufficient features for KNN on {symbol}")
            return None, None, None
        
        # Target: next day's absolute return
        target_col = 'target_abs_return_next_day'
        
        if target_col not in df.columns:
            self.logger.warning(f"Target column not found for {symbol}")
            return None, None, None
        
        # Prepare X and y
        X = df[available_features].copy()
        y = df[target_col].copy()
        
        # Remove rows with NaN
        valid_mask = X.notna().all(axis=1) & y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < self.n_neighbors * 2:
            return None, None, None
        
        # Scale features (important for distance metrics)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save scaler
        scaler_path = self.models_path / f"{symbol}_knn_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        
        return X_scaled, y.values, available_features
    
    def _chronological_split(self, X: np.ndarray, y: np.ndarray, 
                           dates: pd.DatetimeIndex) -> Tuple:
        """Split data chronologically for time-series."""
        split_idx = int(len(X) * (1 - self.test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        train_dates = dates[:split_idx]
        test_dates = dates[split_idx:]
        
        self.logger.info(f"KNN Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test, train_dates, test_dates
    
    def _train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                     symbol: str) -> Dict[str, Any]:
        """Train KNN models for similarity and regression."""
        models = {}
        
        # 1. KNN for similarity search
        self.logger.info(f"Training KNN similarity model for {symbol}")
        similarity_model = NearestNeighbors(
            n_neighbors=self.n_neighbors * 2,  # Get more neighbors for analysis
            metric=self.metric,
            algorithm='auto'
        )
        similarity_model.fit(X_train)
        models['similarity'] = similarity_model
        
        # 2. KNN regressor for prediction
        self.logger.info(f"Training KNN regressor for {symbol}")
        regressor_model = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            weights='distance'  # Weight by inverse distance
        )
        regressor_model.fit(X_train, y_train)
        models['regressor'] = regressor_model
        
        # 3. Additional models with different parameters for comparison
        self.logger.info(f"Training uniform KNN regressor for {symbol}")
        uniform_model = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            weights='uniform'  # Equal weights
        )
        uniform_model.fit(X_train, y_train)
        models['uniform_regressor'] = uniform_model
        
        return models
    
    def _evaluate_models(self, models: Dict[str, Any], X_test: np.ndarray,
                        y_test: np.ndarray, symbol: str) -> Dict[str, Any]:
        """Evaluate KNN regression models."""
        evaluation = {}
        
        # Evaluate distance-weighted KNN
        if 'regressor' in models:
            model = models['regressor']
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            
            evaluation['knn_distance_mse'] = float(mse)
            evaluation['knn_distance_rmse'] = float(rmse)
            evaluation['knn_distance_mae'] = float(mae)
            
            # Calculate correlation
            if len(y_test) > 1:
                corr = np.corrcoef(y_test, y_pred)[0, 1]
                evaluation['knn_distance_correlation'] = float(corr)
        
        # Evaluate uniform KNN
        if 'uniform_regressor' in models:
            model = models['uniform_regressor']
            y_pred_uniform = model.predict(X_test)
            
            mse_uniform = mean_squared_error(y_test, y_pred_uniform)
            rmse_uniform = np.sqrt(mse_uniform)
            
            evaluation['knn_uniform_mse'] = float(mse_uniform)
            evaluation['knn_uniform_rmse'] = float(rmse_uniform)
            
            # Compare performance
            if 'knn_distance_rmse' in evaluation:
                improvement = (evaluation['knn_uniform_rmse'] - evaluation['knn_distance_rmse']) / evaluation['knn_distance_rmse']
                evaluation['distance_vs_uniform_improvement'] = float(improvement * 100)  # Percentage
        
        # Evaluate similarity model quality
        if 'similarity' in models and len(X_test) > 0:
            similarity_model = models['similarity']
            
            # Find neighbors for test points in training data
            distances, indices = similarity_model.kneighbors(X_test, n_neighbors=5)
            
            # Calculate average similarity
            avg_distance = np.mean(distances)
            evaluation['avg_similarity_distance'] = float(avg_distance)
            
            # Calculate consistency of neighbors
            neighbor_consistency = self._calculate_neighbor_consistency(indices)
            evaluation['neighbor_consistency'] = float(neighbor_consistency)
        
        return evaluation
    
    def _calculate_neighbor_consistency(self, indices: np.ndarray) -> float:
        """Calculate how consistent nearest neighbors are across test points."""
        if len(indices) < 2:
            return 0.0
        
        # Count how often each training point appears as a neighbor
        neighbor_counts = {}
        for point_neighbors in indices:
            for neighbor in point_neighbors:
                neighbor_counts[neighbor] = neighbor_counts.get(neighbor, 0) + 1
        
        # Calculate consistency score (higher = more consistent neighbors)
        total_neighbors = len(indices) * indices.shape[1]
        unique_neighbors = len(neighbor_counts)
        
        if total_neighbors > 0:
            consistency = 1 - (unique_neighbors / total_neighbors)
        else:
            consistency = 0.0
        
        return consistency
    
    def _find_historical_analogs(self, model: NearestNeighbors, X: np.ndarray,
                               dates: pd.DatetimeIndex, symbol: str) -> Dict[str, Any]:
        """Find historical analogs for recent patterns."""
        analogs = {}
        
        if len(X) < self.n_neighbors * 2:
            return analogs
        
        # Focus on recent patterns (last 20 days)
        recent_window = min(20, len(X))
        recent_indices = list(range(len(X) - recent_window, len(X)))
        
        analogs['recent_patterns'] = []
        
        for idx in recent_indices[-5:]:  # Analyze last 5 recent patterns
            pattern_date = dates[idx]
            
            # Find nearest neighbors (excluding the pattern itself)
            distances, neighbor_indices = model.kneighbors(
                X[idx:idx+1], 
                n_neighbors=self.n_neighbors * 3  # Get more to filter out recent ones
            )
            
            # Filter to only historical analogs (not too recent)
            historical_neighbors = []
            historical_distances = []
            
            for dist, n_idx in zip(distances[0], neighbor_indices[0]):
                if n_idx != idx and n_idx < idx - 10:  # Exclude self and recent patterns
                    neighbor_date = dates[n_idx]
                    days_apart = (pattern_date - neighbor_date).days
                    
                    historical_neighbors.append({
                        'index': int(n_idx),
                        'date': neighbor_date.strftime('%Y-%m-%d'),
                        'distance': float(dist),
                        'similarity': float(1 / (1 + dist)),  # Convert distance to similarity
                        'days_apart': abs(days_apart)
                    })
            
            # Keep top analogs
            historical_neighbors.sort(key=lambda x: x['distance'])
            top_analogs = historical_neighbors[:self.n_neighbors]
            
            analogs['recent_patterns'].append({
                'pattern_date': pattern_date.strftime('%Y-%m-%d'),
                'pattern_index': int(idx),
                'analogs': top_analogs
            })
        
        # Find best overall historical analog for current pattern
        if recent_indices:
            current_idx = recent_indices[-1]
            distances, neighbor_indices = model.kneighbors(
                X[current_idx:current_idx+1], 
                n_neighbors=10
            )
            
            # Find the best historical analog (excluding recent)
            best_analog = None
            for dist, n_idx in zip(distances[0], neighbor_indices[0]):
                if n_idx != current_idx and n_idx < current_idx - 20:
                    best_analog = {
                        'index': int(n_idx),
                        'date': dates[n_idx].strftime('%Y-%m-%d'),
                        'distance': float(dist),
                        'similarity': float(1 / (1 + dist)),
                        'days_apart': abs((dates[current_idx] - dates[n_idx]).days)
                    }
                    break
            
            if best_analog:
                analogs['best_current_analog'] = best_analog
        
        return analogs
    
    def _calculate_similarities(self, model: NearestNeighbors, X: np.ndarray,
                              symbol: str) -> Dict[str, Any]:
        """Calculate similarity matrix for patterns."""
        similarities = {}
        
        if len(X) > 100:  # Limit size for performance
            sample_size = 50
            step = len(X) // sample_size
            sample_indices = list(range(0, len(X), step))[:sample_size]
            X_sample = X[sample_indices]
        else:
            X_sample = X
            sample_indices = list(range(len(X)))
        
        # Calculate pairwise distances
        distances = model.kneighbors(X_sample, n_neighbors=len(X_sample))
        distance_matrix = distances[0]
        
        # Convert to similarity (0 to 1)
        max_distance = np.max(distance_matrix)
        if max_distance > 0:
            similarity_matrix = 1 - (distance_matrix / max_distance)
        else:
            similarity_matrix = np.ones_like(distance_matrix)
        
        similarities['sample_indices'] = sample_indices
        similarities['distance_matrix'] = distance_matrix.tolist()
        similarities['similarity_matrix'] = similarity_matrix.tolist()
        similarities['avg_similarity'] = float(np.mean(similarity_matrix))
        similarities['similarity_std'] = float(np.std(similarity_matrix))
        
        return similarities
    
    def _make_predictions(self, model: Any, X_test: np.ndarray,
                         y_test: np.ndarray, test_dates: pd.DatetimeIndex,
                         symbol: str) -> Dict[str, Any]:
        """Generate KNN predictions for visualization."""
        predictions = {}
        
        if model is None or len(X_test) == 0:
            return predictions
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get neighbor information for each prediction
        neighbor_info = []
        if hasattr(model, 'kneighbors'):
            distances, indices = model.kneighbors(X_test, n_neighbors=self.n_neighbors)
            
            for i, (pred, actual) in enumerate(zip(y_pred, y_test)):
                neighbor_info.append({
                    'prediction': float(pred),
                    'actual': float(actual),
                    'avg_neighbor_distance': float(np.mean(distances[i])),
                    'neighbor_distance_std': float(np.std(distances[i]))
                })
        
        # Create prediction DataFrame
        pred_df = pd.DataFrame({
            'date': test_dates,
            'actual': y_test,
            'predicted': y_pred,
            'error': y_test - y_pred,
            'abs_error': np.abs(y_test - y_pred)
        })
        
        # Rolling performance
        window = min(20, len(pred_df))
        if window > 0:
            pred_df['rolling_mae'] = pred_df['abs_error'].rolling(window=window).mean()
            pred_df['rolling_correlation'] = pred_df['actual'].rolling(window=window).corr(pred_df['predicted'])
        
        predictions['data'] = pred_df.to_dict('records')
        predictions['neighbor_info'] = neighbor_info[:10]  # Limit for storage
        predictions['summary'] = {
            'mean_actual': float(np.mean(y_test)),
            'mean_predicted': float(np.mean(y_pred)),
            'std_actual': float(np.std(y_test)),
            'std_predicted': float(np.std(y_pred)),
            'correlation': float(np.corrcoef(y_test, y_pred)[0, 1]) if len(y_test) > 1 else 0
        }
        
        return predictions
    
    def _create_knn_charts(self, models: Dict[str, Any], X: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          train_dates: pd.DatetimeIndex, test_dates: pd.DatetimeIndex,
                          analogs: Dict[str, Any], symbol: str,
                          feature_names: List[str]) -> Dict[str, go.Figure]:
        """Create KNN visualization charts."""
        charts = {}
        
        # 1. Historical Analogs Visualization
        if 'recent_patterns' in analogs and analogs['recent_patterns']:
            fig = self._create_analog_chart(analogs, symbol)
            charts['historical_analogs'] = fig
        
        # 2. KNN Prediction Performance
        if 'regressor' in models and len(X_test) > 0:
            fig2 = self._create_prediction_chart(models['regressor'], X_test, y_test, test_dates, symbol)
            charts['knn_predictions'] = fig2
        
        # 3. Similarity Space Visualization
        if len(X) > 10:
            fig3 = self._create_similarity_space_chart(X, symbol)
            charts['similarity_space'] = fig3
        
        # 4. Distance vs Performance Analysis
        if 'regressor' in models and len(X_test) > 0:
            fig4 = self._create_distance_analysis_chart(models['regressor'], X_test, y_test, symbol)
            charts['distance_analysis'] = fig4
        
        return charts
    
    def _create_analog_chart(self, analogs: Dict[str, Any], symbol: str) -> go.Figure:
        """Create chart showing historical analogs."""
        fig = make_subplots(
            rows=2, cols=1,
            vertical_spacing=0.15,
            subplot_titles=(
                f'{symbol}: Current Pattern vs Best Historical Analog',
                'Analog Similarity Distribution'
            )
        )
        
        # Best analog visualization
        if 'best_current_analog' in analogs:
            best_analog = analogs['best_current_analog']
            
            # Placeholder for pattern visualization
            # In a real implementation, you would plot the actual price patterns
            
            fig.add_annotation(
                x=0.5, y=0.7,
                xref="paper", yref="paper",
                text=f"Best Historical Analog Found:<br>"
                     f"Date: {best_analog['date']}<br>"
                     f"Similarity: {best_analog['similarity']:.3f}<br>"
                     f"Days Apart: {best_analog['days_apart']}",
                showarrow=False,
                font=dict(size=14),
                align="center",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                bgcolor="white",
                opacity=0.8
            )
        
        # Similarity distribution
        if 'recent_patterns' in analogs:
            all_similarities = []
            for pattern in analogs['recent_patterns']:
                for analog in pattern['analogs']:
                    all_similarities.append(analog['similarity'])
            
            if all_similarities:
                fig.add_trace(
                    go.Histogram(
                        x=all_similarities,
                        nbinsx=20,
                        name='Analog Similarities',
                        marker_color=self.color_palette['primary'],
                        opacity=0.7,
                        hovertemplate='Similarity: %{x:.3f}<br>Count: %{y}<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                # Add mean line
                mean_sim = np.mean(all_similarities)
                fig.add_vline(
                    x=mean_sim,
                    line_width=2,
                    line_dash="dash",
                    line_color=self.color_palette['warning'],
                    row=2, col=1,
                    annotation_text=f"Mean: {mean_sim:.3f}",
                    annotation_position="top right"
                )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Similarity Score", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        
        return fig
    
    def _create_prediction_chart(self, model: Any, X_test: np.ndarray,
                               y_test: np.ndarray, test_dates: pd.DatetimeIndex,
                               symbol: str) -> go.Figure:
        """Create KNN prediction performance chart."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4],
            subplot_titles=(
                f'{symbol}: KNN Predictions vs Actual',
                'Prediction Error by Distance to Neighbors'
            )
        )
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get neighbor distances
        if hasattr(model, 'kneighbors'):
            distances, _ = model.kneighbors(X_test, n_neighbors=self.n_neighbors)
            avg_distances = np.mean(distances, axis=1)
        else:
            avg_distances = np.zeros(len(X_test))
        
        # Predictions vs Actual
        fig.add_trace(
            go.Scatter(
                x=test_dates,
                y=y_test * 100,
                mode='lines+markers',
                name='Actual',
                line=dict(color=self.color_palette['primary'], width=2),
                marker=dict(size=6),
                hovertemplate='Date: %{x}<br>Actual: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=test_dates,
                y=y_pred * 100,
                mode='lines+markers',
                name='KNN Predicted',
                line=dict(color=self.color_palette['success'], width=2, dash='dash'),
                marker=dict(size=6),
                hovertemplate='Date: %{x}<br>Predicted: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Error by distance
        errors = np.abs(y_test - y_pred) * 100
        
        fig.add_trace(
            go.Scatter(
                x=avg_distances,
                y=errors,
                mode='markers',
                name='Error vs Distance',
                marker=dict(
                    size=10,
                    color=errors,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Error (%)", x=1.1)
                ),
                hovertemplate='Avg Distance: %{x:.3f}<br>Error: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add trend line
        if len(avg_distances) > 1:
            z = np.polyfit(avg_distances, errors, 1)
            p = np.poly1d(z)
            x_range = np.linspace(avg_distances.min(), avg_distances.max(), 100)
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=p(x_range),
                    mode='lines',
                    name='Trend',
                    line=dict(color=self.color_palette['danger'], width=2),
                    hovertemplate='Distance: %{x:.3f}<br>Trend Error: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=700,
            showlegend=True,
            hovermode='closest',
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Average Distance to Neighbors", row=2, col=1)
        fig.update_yaxes(title_text="Absolute Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Prediction Error (%)", row=2, col=1)
        
        return fig
    
    def _create_similarity_space_chart(self, X: np.ndarray, symbol: str) -> go.Figure:
        """Create 2D visualization of similarity space using PCA."""
        from sklearn.decomposition import PCA
        
        fig = go.Figure()
        
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X[:100])  # Limit to 100 points for clarity
        
        # Color by density
        from sklearn.neighbors import KernelDensity
        kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
        kde.fit(X_2d)
        densities = np.exp(kde.score_samples(X_2d))
        
        fig.add_trace(
            go.Scatter(
                x=X_2d[:, 0],
                y=X_2d[:, 1],
                mode='markers',
                name='Market Patterns',
                marker=dict(
                    size=10,
                    color=densities,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Pattern Density")
                ),
                hovertemplate='PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>Density: %{marker.color:.3f}<extra></extra>'
            )
        )
        
        # Highlight recent patterns
        if len(X_2d) > 5:
            fig.add_trace(
                go.Scatter(
                    x=X_2d[-5:, 0],
                    y=X_2d[-5:, 1],
                    mode='markers',
                    name='Recent Patterns',
                    marker=dict(
                        size=15,
                        color=self.color_palette['primary'],
                        symbol='star',
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate='Recent Pattern<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
                )
            )
        
        fig.update_layout(
            title=f'{symbol}: Pattern Similarity Space (PCA Reduced)',
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)",
            height=500,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def _create_distance_analysis_chart(self, model: Any, X_test: np.ndarray,
                                      y_test: np.ndarray, symbol: str) -> go.Figure:
        """Analyze how prediction quality relates to distance."""
        fig = go.Figure()
        
        if not hasattr(model, 'kneighbors'):
            return fig
        
        # Get predictions and neighbor distances
        y_pred = model.predict(X_test)
        distances, _ = model.kneighbors(X_test, n_neighbors=self.n_neighbors)
        avg_distances = np.mean(distances, axis=1)
        errors = np.abs(y_test - y_pred) * 100
        
        # Create 2D histogram
        fig.add_trace(
            go.Histogram2d(
                x=avg_distances,
                y=errors,
                colorscale='Viridis',
                nbinsx=20,
                nbinsy=20,
                colorbar=dict(title="Count"),
                hovertemplate='Distance: %{x:.3f}<br>Error: %{y:.2f}%<br>Count: %{z}<extra></extra>'
            )
        )
        
        # Add trend line
        if len(avg_distances) > 1:
            z = np.polyfit(avg_distances, errors, 1)
            p = np.poly1d(z)
            x_range = np.linspace(avg_distances.min(), avg_distances.max(), 100)
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=p(x_range),
                    mode='lines',
                    name='Error Trend',
                    line=dict(color=self.color_palette['danger'], width=3),
                    hovertemplate='Distance: %{x:.3f}<br>Expected Error: %{y:.2f}%<extra></extra>'
                )
            )
        
        fig.update_layout(
            title=f'{symbol}: Prediction Error vs Neighbor Distance',
            xaxis_title='Average Distance to Neighbors',
            yaxis_title='Prediction Error (%)',
            height=500,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def _generate_knn_insights(self, models: Dict[str, Any], 
                              evaluation: Dict[str, Any],
                              analogs: Dict[str, Any], symbol: str) -> List[str]:
        """Generate insights from KNN analysis."""
        insights = []
        
        # Prediction performance
        rmse = evaluation.get('knn_distance_rmse', 0)
        corr = evaluation.get('knn_distance_correlation', 0)
        
        insights.append(
            f"KNN prediction RMSE: {rmse*100:.2f}%, "
            f"Correlation with actual: {corr:.3f}"
        )
        
        # Distance-weighted vs uniform
        improvement = evaluation.get('distance_vs_uniform_improvement', 0)
        if improvement > 5:
            insights.append(f"Distance weighting improves predictions by {improvement:.1f}% over uniform weighting.")
        
        # Similarity analysis
        avg_distance = evaluation.get('avg_similarity_distance', 0)
        if avg_distance > 1:
            insights.append(f"Historical patterns show moderate similarity (avg distance: {avg_distance:.2f}).")
        else:
            insights.append(f"Historical patterns show strong similarity (avg distance: {avg_distance:.2f}).")
        
        # Historical analogs
        if 'best_current_analog' in analogs:
            analog = analogs['best_current_analog']
            insights.append(
                f"Best historical analog found from {analog['date']} "
                f"(similarity: {analog['similarity']:.3f}, {analog['days_apart']} days apart)."
            )
        
        # Neighbor consistency
        consistency = evaluation.get('neighbor_consistency', 0)
        if consistency > 0.3:
            insights.append(f"Strong neighbor consistency ({consistency:.2f}), suggesting recurring patterns.")
        
        return insights
    
    def _save_models(self, models: Dict[str, Any], symbol: str):
        """Save trained KNN models to disk."""
        for name, model in models.items():
            model_path = self.models_path / f"{symbol}_knn_{name}.pkl"
            joblib.dump(model, model_path)
        
        self.logger.debug(f"Saved KNN models for {symbol}")
    
    def _analyze_cross_symbol_knn(self, results: Dict[str, Any], 
                                data_by_symbol: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze KNN results across symbols."""
        cross_results = {}
        
        symbols = [s for s in results.keys() if s != 'cross_symbol']
        
        if len(symbols) < 2:
            return cross_results
        
        # Collect performance metrics
        performance_data = []
        
        for symbol in symbols:
            symbol_data = results[symbol]
            metrics = symbol_data.get('metrics', {})
            
            rmse = metrics.get('knn_distance_rmse', 0)
            corr = metrics.get('knn_distance_correlation', 0)
            avg_distance = metrics.get('avg_similarity_distance', 0)
            
            performance_data.append({
                'symbol': symbol,
                'rmse': rmse,
                'correlation': corr,
                'avg_distance': avg_distance
            })
        
        # Create comparison chart
        fig = go.Figure()
        
        symbols_list = [p['symbol'] for p in performance_data]
        rmse_values = [p['rmse'] * 100 for p in performance_data]  # Convert to percentage
        corr_values = [p['correlation'] for p in performance_data]
        
        # RMSE bars
        fig.add_trace(
            go.Bar(
                x=symbols_list,
                y=rmse_values,
                name='RMSE (%)',
                marker_color=self.color_palette['primary'],
                text=[f'{v:.2f}%' for v in rmse_values],
                textposition='auto',
                hovertemplate='Symbol: %{x}<br>RMSE: %{y:.2f}%<extra></extra>'
            )
        )
        
        # Correlation line
        fig.add_trace(
            go.Scatter(
                x=symbols_list,
                y=corr_values,
                name='Correlation',
                mode='lines+markers',
                yaxis='y2',
                line=dict(color=self.color_palette['success'], width=3),
                marker=dict(size=10),
                text=[f'{v:.3f}' for v in corr_values],
                hovertemplate='Correlation: %{y:.3f}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title='Cross-Symbol KNN Performance Comparison',
            height=500,
            showlegend=True,
            template='plotly_white',
            yaxis=dict(
                title='RMSE (%)',
                titlefont=dict(color=self.color_palette['primary']),
                tickfont=dict(color=self.color_palette['primary'])
            ),
            yaxis2=dict(
                title='Correlation',
                titlefont=dict(color=self.color_palette['success']),
                tickfont=dict(color=self.color_palette['success']),
                overlaying='y',
                side='right'
            )
        )
        
        cross_results['performance_comparison_chart'] = fig
        
        # Generate insights
        insights = []
        
        if performance_data:
            # Find best and worst predictions
            best_by_rmse = min(performance_data, key=lambda x: x['rmse'])
            worst_by_rmse = max(performance_data, key=lambda x: x['rmse'])
            
            insights.append(
                f"Best KNN predictions: {best_by_rmse['symbol']} "
                f"(RMSE: {best_by_rmse['rmse']*100:.2f}%)"
            )
            insights.append(
                f"Most challenging for KNN: {worst_by_rmse['symbol']} "
                f"(RMSE: {worst_by_rmse['rmse']*100:.2f}%)"
            )
            
            # Pattern similarity across symbols
            avg_distances = [p['avg_distance'] for p in performance_data]
            if len(avg_distances) > 0:
                avg_overall_distance = np.mean(avg_distances)
                insights.append(f"Average pattern distance across symbols: {avg_overall_distance:.3f}")
        
        cross_results['insights'] = insights
        
        return cross_results
    
    def _save_symbol_results(self, symbol: str, results: Dict[str, Any]):
        """Save analysis results to artifacts folder for dashboard."""
        try:
            # Remove model objects before saving (already saved separately)
            save_results = results.copy()
            
            # Remove sklearn model objects
            if 'models' in save_results:
                del save_results['models']
            
            # Save to artifacts folder
            artifact_path = self.artifacts_path / f"{symbol}_knn_models.pkl"
            joblib.dump(save_results, artifact_path)
            
            self.logger.info(f"Saved KNN results for {symbol} to {artifact_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving KNN results for {symbol}: {str(e)}")