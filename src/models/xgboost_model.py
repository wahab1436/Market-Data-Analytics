"""
XGBoost Model Module
Gradient boosting for next-day absolute return prediction with time-series validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class XGBoostModel:
    """XGBoost model for volatility prediction with rigorous time-series validation."""
    
    def __init__(self, config: Dict[str, Any], logger):
        """Initialize XGBoost modeler."""
        self.config = config
        self.logger = logger
        self.color_palette = config['dashboard']['color_palette']
        self.models_path = Path(config['paths']['models'])
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize artifacts path for saving results
        self.artifacts_path = Path(config['paths']['artifacts'])
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # XGBoost parameters
        xgb_params = config['models']['xgboost']
        self.params = {
            'n_estimators': xgb_params['n_estimators'],
            'max_depth': xgb_params['max_depth'],
            'learning_rate': xgb_params['learning_rate'],
            'objective': xgb_params['objective'],
            'eval_metric': xgb_params['eval_metric'],
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # Validation parameters
        self.test_size = config['models']['validation']['test_size']
        self.train_test_split = pd.to_datetime(config['models']['validation']['train_test_split'])
    
    def train_and_evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Train and evaluate XGBoost model with time-series validation."""
        results = {}
        
        by_symbol = data['by_symbol']
        
        for symbol, df in by_symbol.items():
            self.logger.info(f"Training XGBoost model for {symbol}")
            
            symbol_results = {
                'charts': {},
                'insights': [],
                'metrics': {},
                'model': None,
                'feature_importance': {},
                'predictions': {}
            }
            
            # Prepare data for XGBoost
            X, y, feature_names = self._prepare_xgboost_data(df, symbol)
            
            if X is None or len(X) < 50:
                self.logger.warning(f"Insufficient data for XGBoost on {symbol}")
                continue
            
            # Split data chronologically
            X_train, X_test, y_train, y_test, train_dates, test_dates = self._chronological_split(
                X, y, df.index
            )
            
            # Train XGBoost model
            model, train_metrics = self._train_model(X_train, y_train, X_test, y_test, symbol)
            symbol_results['model'] = model
            symbol_results['train_metrics'] = train_metrics
            
            # Evaluate model
            evaluation = self._evaluate_model(model, X_test, y_test, symbol)
            symbol_results['metrics'].update(evaluation)
            
            # Get feature importance
            importance = self._get_feature_importance(model, feature_names, symbol)
            symbol_results['feature_importance'] = importance
            
            # Make predictions
            predictions = self._make_predictions(model, X_train, X_test, y_train, y_test, 
                                                train_dates, test_dates, symbol)
            symbol_results['predictions'] = predictions
            
            # Create XGBoost charts
            xgb_charts = self._create_xgboost_charts(
                model, X_train, X_test, y_train, y_test, 
                train_dates, test_dates, importance, symbol, feature_names
            )
            symbol_results['charts'].update(xgb_charts)
            
            # Generate insights
            insights = self._generate_xgboost_insights(model, evaluation, importance, symbol)
            symbol_results['insights'].extend(insights)
            
            # Save model and data for explainability
            self._save_model_data(model, X_train, X_test, y_train, y_test, 
                                 train_dates, test_dates, feature_names, symbol)
            
            results[symbol] = symbol_results
            
            # FIXED: Save results to unified model results file
            self._save_symbol_results(symbol, symbol_results)
        
        # Cross-symbol XGBoost comparison
        if len(by_symbol) > 1:
            cross_results = self._analyze_cross_symbol_xgboost(results)
            results['cross_symbol'] = cross_results
        
        return results
    
    def _prepare_xgboost_data(self, df: pd.DataFrame, symbol: str) -> Tuple:
        """Prepare data for XGBoost model."""
        # Define comprehensive feature set
        feature_categories = [
            # Returns and momentum
            'return', 'log_return',
            'return_lag_1d', 'return_lag_5d', 'return_lag_10d',
            'rolling_return_10d', 'rolling_return_20d', 'rolling_return_30d',
            'momentum_10d', 'roc_10d',
            
            # Volatility features
            'volatility_10d', 'volatility_20d', 'volatility_30d',
            'volatility_annualized_10d', 'volatility_annualized_20d',
            'daily_range_pct', 'atr_14d',
            
            # Volume features
            'volume_ratio_10d', 'volume_ratio_20d', 'volume_ratio_30d',
            'volume_std_20d', 'volume_return_corr_20d',
            'dollar_volume_ma_20d',
            
            # Price structure
            'ma_20d', 'ma_50d', 'ma_200d',
            'price_vs_ma_20d_pct', 'price_vs_ma_50d_pct', 'price_vs_ma_200d_pct',
            'rolling_high_20d', 'rolling_low_20d',
            'price_vs_high_20d', 'price_vs_low_20d',
            'price_position'
        ]
        
        # Filter out None values and select available features
        available_features = [f for f in feature_categories if f is not None and f in df.columns]
        
        if len(available_features) < 10:
            self.logger.warning(f"Insufficient features for XGBoost on {symbol}")
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
        
        if len(X) < 50:
            return None, None, None
        
        # Note: XGBoost handles feature scaling internally, but we'll scale for consistency
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save scaler
        scaler_path = self.models_path / f"{symbol}_xgboost_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        
        return X_scaled, y.values, available_features
    
    def _chronological_split(self, X: np.ndarray, y: np.ndarray, 
                            dates: pd.DatetimeIndex) -> Tuple:
        """Split data chronologically for time-series validation."""
        # Align dates with valid data
        split_idx = int(len(X) * (1 - self.test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        train_dates = dates[:split_idx]
        test_dates = dates[split_idx:]
        
        self.logger.info(f"XGBoost Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test, train_dates, test_dates
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray, y_test: np.ndarray, 
                    symbol: str) -> Tuple[xgb.Booster, Dict[str, float]]:
        """Train XGBoost model with early stopping."""
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Training parameters with early stopping
        evals = [(dtrain, 'train'), (dtest, 'test')]
        
        # Train model
        model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params['n_estimators'],
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Get training metrics
        train_pred = model.predict(dtrain)
        train_metrics = {
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, train_pred))),
            'train_mae': float(mean_absolute_error(y_train, train_pred)),
            'train_r2': float(r2_score(y_train, train_pred))
        }
        
        self.logger.info(
            f"XGBoost training complete for {symbol}: "
            f"R²={train_metrics['train_r2']:.3f}, "
            f"RMSE={train_metrics['train_rmse']*100:.2f}%"
        )
        
        return model, train_metrics
    
    def _evaluate_model(self, model: xgb.Booster, X_test: np.ndarray, 
                       y_test: np.ndarray, symbol: str) -> Dict[str, float]:
        """Evaluate model performance on test set."""
        dtest = xgb.DMatrix(X_test)
        y_pred = model.predict(dtest)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Correlation
        correlation = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else 0
        
        # Error analysis
        errors = y_pred - y_test
        error_mean = np.mean(errors)
        error_std = np.std(errors)
        
        # Prediction intervals
        within_1std = np.mean(np.abs(errors) <= error_std) * 100
        within_2std = np.mean(np.abs(errors) <= 2 * error_std) * 100
        
        # Direction accuracy (for sign of return)
        direction_accuracy = np.mean(np.sign(y_pred) == np.sign(y_test)) * 100
        
        metrics = {
            'test_rmse': float(rmse),
            'test_mae': float(mae),
            'test_r2': float(r2),
            'test_correlation': float(correlation),
            'error_mean': float(error_mean),
            'error_std': float(error_std),
            'within_1std_pct': float(within_1std),
            'within_2std_pct': float(within_2std),
            'direction_accuracy': float(direction_accuracy)
        }
        
        self.logger.info(
            f"XGBoost evaluation for {symbol}: "
            f"Test R²={r2:.3f}, RMSE={rmse*100:.2f}%, "
            f"Correlation={correlation:.3f}"
        )
        
        return metrics
    
    def _get_feature_importance(self, model: xgb.Booster, 
                               feature_names: List[str], 
                               symbol: str) -> Dict[str, Any]:
        """Get feature importance from XGBoost model."""
        # Get importance by gain (most informative)
        importance_gain = model.get_score(importance_type='gain')
        importance_weight = model.get_score(importance_type='weight')
        importance_cover = model.get_score(importance_type='cover')
        
        # Convert to list of dicts
        importance_list = []
        for i, fname in enumerate(feature_names):
            feature_key = f'f{i}'
            if feature_key in importance_gain:
                importance_list.append({
                    'feature': fname,
                    'gain': float(importance_gain.get(feature_key, 0)),
                    'weight': float(importance_weight.get(feature_key, 0)),
                    'cover': float(importance_cover.get(feature_key, 0))
                })
        
        # Sort by gain
        importance_list.sort(key=lambda x: x['gain'], reverse=True)
        
        # Normalize
        total_gain = sum(item['gain'] for item in importance_list)
        if total_gain > 0:
            for item in importance_list:
                item['gain_pct'] = float((item['gain'] / total_gain) * 100)
        
        # Group by feature type
        feature_types = {
            'returns': ['return', 'log_return', 'rolling_return', 'momentum', 'roc'],
            'volatility': ['volatility', 'atr', 'range'],
            'volume': ['volume', 'dollar_volume'],
            'price_structure': ['ma', 'price_vs', 'rolling_high', 'rolling_low', 'position']
        }
        
        importance_by_type = {ftype: 0.0 for ftype in feature_types}
        
        for item in importance_list:
            fname = item['feature']
            for ftype, keywords in feature_types.items():
                if any(kw in fname.lower() for kw in keywords):
                    importance_by_type[ftype] += item.get('gain_pct', 0)
                    break
        
        return {
            'by_gain': importance_list[:20],  # Top 20 features
            'by_type': importance_by_type
        }
    
    def _make_predictions(self, model: xgb.Booster, 
                         X_train: np.ndarray, X_test: np.ndarray,
                         y_train: np.ndarray, y_test: np.ndarray,
                         train_dates: pd.DatetimeIndex, test_dates: pd.DatetimeIndex,
                         symbol: str) -> Dict[str, Any]:
        """Make predictions and organize results for dashboard."""
        dtrain = xgb.DMatrix(X_train)
        dtest = xgb.DMatrix(X_test)
        
        train_pred = model.predict(dtrain)
        test_pred = model.predict(dtest)
        
        predictions = {
            'train': {
                'dates': train_dates.tolist(),
                'actual': y_train.tolist(),
                'predicted': train_pred.tolist(),
                'errors': (train_pred - y_train).tolist()
            },
            'test': {
                'dates': test_dates.tolist(),
                'actual': y_test.tolist(),
                'predicted': test_pred.tolist(),
                'errors': (test_pred - y_test).tolist()
            }
        }
        
        return predictions
    
    def _create_xgboost_charts(self, model: xgb.Booster, 
                              X_train: np.ndarray, X_test: np.ndarray,
                              y_train: np.ndarray, y_test: np.ndarray,
                              train_dates: pd.DatetimeIndex, test_dates: pd.DatetimeIndex,
                              importance: Dict[str, Any], symbol: str,
                              feature_names: List[str]) -> Dict[str, go.Figure]:
        """Create comprehensive XGBoost visualization charts."""
        charts = {}
        
        # 1. Predictions vs Actuals (Time Series)
        dtrain = xgb.DMatrix(X_train)
        dtest = xgb.DMatrix(X_test)
        train_pred = model.predict(dtrain)
        test_pred = model.predict(dtest)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                f'{symbol} - XGBoost Predictions vs Actuals (Training Period)',
                f'{symbol} - XGBoost Predictions vs Actuals (Test Period)'
            ),
            vertical_spacing=0.12
        )
        
        # Training period
        fig.add_trace(
            go.Scatter(
                x=train_dates,
                y=y_train * 100,
                name='Actual',
                line=dict(color=self.color_palette['text'], width=1),
                opacity=0.7,
                hovertemplate='Date: %{x}<br>Actual: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=train_dates,
                y=train_pred * 100,
                name='Predicted',
                line=dict(color=self.color_palette['primary'], width=2),
                hovertemplate='Date: %{x}<br>Predicted: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Test period
        fig.add_trace(
            go.Scatter(
                x=test_dates,
                y=y_test * 100,
                name='Actual',
                line=dict(color=self.color_palette['text'], width=1),
                opacity=0.7,
                showlegend=False,
                hovertemplate='Date: %{x}<br>Actual: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=test_dates,
                y=test_pred * 100,
                name='Predicted',
                line=dict(color=self.color_palette['warning'], width=2),
                showlegend=False,
                hovertemplate='Date: %{x}<br>Predicted: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Next-Day Volatility (%)", row=1, col=1)
        fig.update_yaxes(title_text="Next-Day Volatility (%)", row=2, col=1)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            template='plotly_white',
            hovermode='x unified'
        )
        
        charts['predictions_timeseries'] = fig
        
        # 2. Scatter Plot: Predicted vs Actual
        fig2 = go.Figure()
        
        # Test points
        fig2.add_trace(
            go.Scatter(
                x=y_test * 100,
                y=test_pred * 100,
                mode='markers',
                name='Test Set',
                marker=dict(
                    color=self.color_palette['warning'],
                    size=8,
                    opacity=0.6,
                    line=dict(width=0.5, color='white')
                ),
                hovertemplate='Actual: %{x:.2f}%<br>Predicted: %{y:.2f}%<extra></extra>'
            )
        )
        
        # Perfect prediction line
        min_val = min(y_test.min(), test_pred.min()) * 100
        max_val = max(y_test.max(), test_pred.max()) * 100
        
        fig2.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color=self.color_palette['text'], dash='dash', width=2),
                showlegend=True,
                hoverinfo='skip'
            )
        )
        
        # Calculate R²
        r2 = r2_score(y_test, test_pred)
        
        fig2.update_layout(
            title=f'{symbol} - XGBoost: Predicted vs Actual (R² = {r2:.3f})',
            xaxis_title='Actual Next-Day Volatility (%)',
            yaxis_title='Predicted Next-Day Volatility (%)',
            height=600,
            template='plotly_white',
            showlegend=True
        )
        
        charts['predicted_vs_actual'] = fig2
        
        # 3. Feature Importance Chart
        if 'by_gain' in importance and importance['by_gain']:
            top_n = min(15, len(importance['by_gain']))
            top_features = importance['by_gain'][:top_n]
            
            fig3 = go.Figure()
            
            features = [f['feature'] for f in top_features]
            gains = [f.get('gain_pct', 0) for f in top_features]
            
            fig3.add_trace(
                go.Bar(
                    y=features[::-1],  # Reverse for better readability
                    x=gains[::-1],
                    orientation='h',
                    marker=dict(
                        color=gains[::-1],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Importance %')
                    ),
                    text=[f'{g:.1f}%' for g in gains[::-1]],
                    textposition='auto',
                    hovertemplate='%{y}<br>Importance: %{x:.1f}%<extra></extra>'
                )
            )
            
            fig3.update_layout(
                title=f'{symbol} - XGBoost Feature Importance (by Gain)',
                xaxis_title='Importance (%)',
                yaxis_title='Feature',
                height=500,
                template='plotly_white',
                showlegend=False
            )
            
            charts['feature_importance'] = fig3
        
        # 4. Residual Analysis
        errors = test_pred - y_test
        
        fig4 = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Residual Distribution', 'Residuals Over Time'),
            specs=[[{'type': 'histogram'}, {'type': 'scatter'}]]
        )
        
        # Histogram
        fig4.add_trace(
            go.Histogram(
                x=errors * 100,
                nbinsx=30,
                name='Residuals',
                marker_color=self.color_palette['primary'],
                opacity=0.7,
                hovertemplate='Error: %{x:.2f}%<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Time series
        fig4.add_trace(
            go.Scatter(
                x=test_dates,
                y=errors * 100,
                mode='markers',
                name='Residuals',
                marker=dict(
                    color=self.color_palette['warning'],
                    size=6,
                    opacity=0.6
                ),
                hovertemplate='Date: %{x}<br>Error: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Zero line
        fig4.add_hline(
            y=0, line_dash="dash", line_color=self.color_palette['text'],
            row=1, col=2
        )
        
        fig4.update_xaxes(title_text="Prediction Error (%)", row=1, col=1)
        fig4.update_xaxes(title_text="Date", row=1, col=2)
        fig4.update_yaxes(title_text="Frequency", row=1, col=1)
        fig4.update_yaxes(title_text="Prediction Error (%)", row=1, col=2)
        
        fig4.update_layout(
            title=f'{symbol} - XGBoost Residual Analysis',
            height=500,
            showlegend=False,
            template='plotly_white'
        )
        
        charts['residual_analysis'] = fig4
        
        return charts
    
    def _generate_xgboost_insights(self, model: xgb.Booster, 
                                  evaluation: Dict[str, float],
                                  importance: Dict[str, Any],
                                  symbol: str) -> List[str]:
        """Generate actionable insights from XGBoost model."""
        insights = []
        
        # Model performance
        r2 = evaluation.get('test_r2', 0)
        rmse = evaluation.get('test_rmse', 0)
        corr = evaluation.get('test_correlation', 0)
        
        if r2 > 0.5:
            insights.append(f"Strong XGBoost model: R² = {r2:.3f}, explaining {r2*100:.1f}% of volatility variance.")
        elif r2 > 0.2:
            insights.append("Moderate predictive power for next-day volatility.")
        else:
            insights.append("Limited predictive power, suggesting highly stochastic volatility.")
        
        # Error analysis
        within_1std = evaluation.get('within_1std_pct', 0)
        if within_1std > 70:
            insights.append(f"Most predictions within 1 standard deviation ({within_1std:.1f}%).")
        
        error_mean = evaluation.get('error_mean', 0)
        if abs(error_mean) > 0.01:
            direction = "under" if error_mean > 0 else "over"
            insights.append(f"Model tends to {direction}predict volatility (bias: {error_mean*100:.2f}%).")
        
        # Feature importance insights
        if 'by_gain' in importance and importance['by_gain']:
            top_features = importance['by_gain'][:3]
            top_feature_names = [f['feature'] for f in top_features]
            insights.append(f"Most important features: {', '.join(top_feature_names)}.")
            
            # Check feature categories
            if 'by_type' in importance:
                type_importance = importance['by_type']
                top_type = max(type_importance.items(), key=lambda x: x[1])
                insights.append(f"Most predictive feature type: {top_type[0]} ({top_type[1]:.1f}% importance).")
        
        # Model complexity
        if hasattr(model, 'best_iteration'):
            insights.append(f"Optimal model complexity reached at {model.best_iteration} boosting rounds.")
        
        return insights
    
    def _save_model_data(self, model: xgb.Booster, X_train: np.ndarray, 
                        X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                        train_dates: pd.DatetimeIndex, test_dates: pd.DatetimeIndex,
                        feature_names: List[str], symbol: str):
        """Save model and data for explainability and reuse."""
        # Save XGBoost model
        model_path = self.models_path / f"{symbol}_xgboost_model.json"
        model.save_model(str(model_path))
        
        # Save feature names
        features_path = self.models_path / f"{symbol}_xgboost_features.pkl"
        joblib.dump(feature_names, features_path)
        
        # Save test data for explainability
        explain_data = {
            'X_test': X_test,
            'y_test': y_test,
            'test_dates': test_dates,
            'feature_names': feature_names
        }
        explain_path = self.models_path / f"{symbol}_xgboost_explain_data.pkl"
        joblib.dump(explain_data, explain_path)
        
        self.logger.debug(f"Saved XGBoost model and data for {symbol}")
    
    def _save_symbol_results(self, symbol: str, results: Dict[str, Any]):
        """
        FIXED: Save XGBoost results to unified model results file.
        Converts Plotly figures to JSON and saves only serializable data.
        """
        try:
            # Convert Plotly figures to JSON
            if 'charts' in results:
                serialized_charts = {}
                for chart_name, fig in results['charts'].items():
                    if isinstance(fig, go.Figure):
                        serialized_charts[chart_name] = fig.to_json()
                    else:
                        serialized_charts[chart_name] = fig
                results['charts'] = serialized_charts
            
            # Remove non-serializable objects
            results.pop('model', None)
            results.pop('explainability_data', None)
            
            # Load existing model results or create new
            model_file = self.artifacts_path / f"{symbol}_model_results.pkl"
            if model_file.exists():
                existing_data = joblib.load(model_file)
            else:
                existing_data = {}
            
            # Update with XGBoost results
            existing_data['xgboost'] = results
            
            # Save back to unified file
            joblib.dump(existing_data, model_file)
            
            self.logger.info(f"Saved XGBoost results for {symbol} to {model_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving XGBoost results for {symbol}: {e}")
    
    def _analyze_cross_symbol_xgboost(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze XGBoost results across symbols."""
        cross_results = {}
        
        symbols = [s for s in results.keys() if s != 'cross_symbol']
        
        if len(symbols) < 2:
            return cross_results
        
        # Collect performance metrics
        performance_data = []
        
        for symbol in symbols:
            symbol_data = results[symbol]
            metrics = symbol_data.get('metrics', {})
            
            rmse = metrics.get('test_rmse', 0)
            r2 = metrics.get('test_r2', 0)
            corr = metrics.get('test_correlation', 0)
            
            performance_data.append({
                'symbol': symbol,
                'rmse': rmse,
                'r2': r2,
                'correlation': corr
            })
        
        # Create comparison chart
        fig = go.Figure()
        
        symbols_list = [p['symbol'] for p in performance_data]
        rmse_values = [p['rmse'] * 100 for p in performance_data]
        r2_values = [p['r2'] for p in performance_data]
        
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
        
        # R² line
        fig.add_trace(
            go.Scatter(
                x=symbols_list,
                y=r2_values,
                name='R² Score',
                mode='lines+markers',
                yaxis='y2',
                line=dict(color=self.color_palette['success'], width=3),
                marker=dict(size=10),
                text=[f'{v:.3f}' for v in r2_values],
                hovertemplate='R²: %{y:.3f}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title='Cross-Symbol XGBoost Performance Comparison',
            height=500,
            showlegend=True,
            template='plotly_white',
            yaxis=dict(
                title='RMSE (%)',
                titlefont=dict(color=self.color_palette['primary']),
                tickfont=dict(color=self.color_palette['primary'])
            ),
            yaxis2=dict(
                title='R² Score',
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
            best_by_r2 = max(performance_data, key=lambda x: x['r2'])
            worst_by_r2 = min(performance_data, key=lambda x: x['r2'])
            
            insights.append(
                f"Best XGBoost predictions: {best_by_r2['symbol']} "
                f"(R²: {best_by_r2['r2']:.3f}, RMSE: {best_by_r2['rmse']*100:.2f}%)"
            )
            insights.append(
                f"Most challenging for XGBoost: {worst_by_r2['symbol']} "
                f"(R²: {worst_by_r2['r2']:.3f}, RMSE: {worst_by_r2['rmse']*100:.2f}%)"
            )
            
            # Check consistency
            avg_r2 = np.mean([p['r2'] for p in performance_data])
            r2_std = np.std([p['r2'] for p in performance_data])
            
            if r2_std < 0.1:
                insights.append(f"Consistent performance across symbols (R² std: {r2_std:.3f}).")
            else:
                insights.append(f"Variable performance across symbols (R² std: {r2_std:.3f}).")
        
        cross_results['insights'] = insights
        
        return cross_results
