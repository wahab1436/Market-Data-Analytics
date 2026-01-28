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
            predictions = self._make_predictions(model, X_test, y_test, test_dates, symbol)
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
            
            # Store data needed for explainability
            symbol_results['explainability_data'] = {
                'model': model,
                'X_test': X_test,
                'feature_names': feature_names,
                'test_dates': test_dates
            }
            
            results[symbol] = symbol_results
        
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
            'volatility_zscore_20d' if 'volatility_zscore_20d' in df.columns else None,
            
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
                    X_val: np.ndarray, y_val: np.ndarray, symbol: str) -> Tuple:
        """Train XGBoost model with early stopping."""
        self.logger.info(f"Training XGBoost for {symbol} with {X_train.shape[1]} features")
        
        # Convert to DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Set up early stopping
        early_stopping_rounds = 20
        
        # Train model
        model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.params['n_estimators'],
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )
        
        # Get training metrics
        train_pred = model.predict(dtrain)
        val_pred = model.predict(dval)
        
        train_metrics = {
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, train_pred))),
            'train_mae': float(mean_absolute_error(y_train, train_pred)),
            'train_r2': float(r2_score(y_train, train_pred)),
            'val_rmse': float(np.sqrt(mean_squared_error(y_val, val_pred))),
            'val_mae': float(mean_absolute_error(y_val, val_pred)),
            'val_r2': float(r2_score(y_val, val_pred)),
            'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else self.params['n_estimators']
        }
        
        self.logger.info(f"XGBoost trained for {symbol}, best iteration: {train_metrics['best_iteration']}")
        
        return model, train_metrics
    
    def _evaluate_model(self, model: xgb.Booster, X_test: np.ndarray,
                       y_test: np.ndarray, symbol: str) -> Dict[str, Any]:
        """Evaluate XGBoost model performance."""
        evaluation = {}
        
        # Make predictions
        dtest = xgb.DMatrix(X_test)
        y_pred = model.predict(dtest)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        evaluation['test_mse'] = float(mse)
        evaluation['test_rmse'] = float(rmse)
        evaluation['test_mae'] = float(mae)
        evaluation['test_r2'] = float(r2)
        
        # Direction accuracy (for volatility prediction)
        # We'll use correlation as a measure of directional accuracy
        if len(y_test) > 1:
            corr = np.corrcoef(y_test, y_pred)[0, 1]
            evaluation['test_correlation'] = float(corr)
        
        # Error distribution metrics
        errors = y_test - y_pred
        evaluation['error_mean'] = float(np.mean(errors))
        evaluation['error_std'] = float(np.std(errors))
        evaluation['error_skew'] = float(pd.Series(errors).skew())
        
        # Percent within 1 standard deviation of actual
        std_actual = np.std(y_test)
        within_1std = (np.abs(errors) < std_actual).mean()
        evaluation['within_1std_pct'] = float(within_1std * 100)
        
        # Quantile analysis of errors
        error_quantiles = np.percentile(np.abs(errors), [25, 50, 75, 90])
        evaluation['error_q25'] = float(error_quantiles[0])
        evaluation['error_median'] = float(error_quantiles[1])
        evaluation['error_q75'] = float(error_quantiles[2])
        evaluation['error_q90'] = float(error_quantiles[3])
        
        self.logger.info(f"XGBoost evaluation for {symbol}: RMSE={rmse*100:.2f}%, R²={r2:.3f}")
        
        return evaluation
    
    def _get_feature_importance(self, model: xgb.Booster, 
                               feature_names: List[str], symbol: str) -> Dict[str, Any]:
        """Get feature importance from XGBoost model."""
        importance = {}
        
        # Get importance scores
        importance_scores = model.get_score(importance_type='gain')
        
        if not importance_scores:
            # Try weight importance
            importance_scores = model.get_score(importance_type='weight')
        
        if importance_scores:
            # Convert to list with feature names
            importance_list = []
            for i, feature in enumerate(feature_names):
                # XGBoost uses f0, f1, f2... as feature names
                feature_key = f'f{i}'
                score = importance_scores.get(feature_key, 0)
                importance_list.append({
                    'feature': feature,
                    'importance': float(score),
                    'importance_normalized': 0  # Will be calculated below
                })
            
            # Normalize importance scores
            total_importance = sum(item['importance'] for item in importance_list)
            if total_importance > 0:
                for item in importance_list:
                    item['importance_normalized'] = float(item['importance'] / total_importance * 100)
            
            # Sort by importance
            importance_list.sort(key=lambda x: x['importance'], reverse=True)
            
            importance['by_gain'] = importance_list[:20]  # Top 20 features
            
            # Calculate importance by type
            importance_by_type = self._categorize_feature_importance(importance_list)
            importance['by_type'] = importance_by_type
        else:
            self.logger.warning(f"Could not extract feature importance for {symbol}")
        
        return importance
    
    def _categorize_feature_importance(self, importance_list: List[Dict]) -> Dict[str, float]:
        """Categorize features by type for aggregated importance."""
        categories = {
            'returns_momentum': ['return', 'log_return', 'rolling_return', 'momentum', 'roc'],
            'volatility': ['volatility', 'range', 'atr', 'zscore'],
            'volume': ['volume', 'dollar_volume'],
            'price_structure': ['ma_', 'price_vs_ma', 'rolling_high', 'rolling_low', 'price_position'],
            'lagged': ['lag_', 'return_lag']
        }
        
        category_importance = {cat: 0.0 for cat in categories.keys()}
        category_counts = {cat: 0 for cat in categories.keys()}
        
        for item in importance_list:
            feature = item['feature']
            importance = item['importance_normalized']
            
            for cat, patterns in categories.items():
                if any(pattern in feature for pattern in patterns):
                    category_importance[cat] += importance
                    category_counts[cat] += 1
        
        # Normalize category importance
        total = sum(category_importance.values())
        if total > 0:
            for cat in category_importance:
                category_importance[cat] = category_importance[cat] / total * 100
        
        return category_importance
    
    def _make_predictions(self, model: xgb.Booster, X_test: np.ndarray,
                         y_test: np.ndarray, test_dates: pd.DatetimeIndex,
                         symbol: str) -> Dict[str, Any]:
        """Generate XGBoost predictions for visualization."""
        predictions = {}
        
        # Make predictions
        dtest = xgb.DMatrix(X_test)
        y_pred = model.predict(dtest)
        
        # Create prediction DataFrame
        pred_df = pd.DataFrame({
            'date': test_dates,
            'actual': y_test,
            'predicted': y_pred,
            'error': y_test - y_pred,
            'abs_error': np.abs(y_test - y_pred),
            'error_pct': (y_test - y_pred) / (y_test + 1e-8) * 100
        })
        
        # Rolling performance metrics
        window = min(20, len(pred_df))
        if window > 0:
            pred_df['rolling_mae'] = pred_df['abs_error'].rolling(window=window).mean()
            pred_df['rolling_correlation'] = pred_df['actual'].rolling(window=window).corr(pred_df['predicted'])
            pred_df['rolling_r2'] = 1 - (pred_df['abs_error'].rolling(window=window).var() / 
                                        pred_df['actual'].rolling(window=window).var())
        
        predictions['data'] = pred_df.to_dict('records')
        predictions['summary'] = {
            'mean_actual': float(np.mean(y_test)),
            'mean_predicted': float(np.mean(y_pred)),
            'std_actual': float(np.std(y_test)),
            'std_predicted': float(np.std(y_pred)),
            'correlation': float(np.corrcoef(y_test, y_pred)[0, 1]) if len(y_test) > 1 else 0,
            'bias': float(np.mean(y_test - y_pred)),  # Positive means underprediction
            'mse': float(mean_squared_error(y_test, y_pred)),
            'mae': float(mean_absolute_error(y_test, y_pred))
        }
        
        # Identify best and worst predictions
        if len(pred_df) > 0:
            best_idx = pred_df['abs_error'].idxmin()
            worst_idx = pred_df['abs_error'].idxmax()
            
            predictions['best_prediction'] = {
                'date': pred_df.loc[best_idx, 'date'].strftime('%Y-%m-%d'),
                'actual': float(pred_df.loc[best_idx, 'actual']),
                'predicted': float(pred_df.loc[best_idx, 'predicted']),
                'error': float(pred_df.loc[best_idx, 'error'])
            }
            
            predictions['worst_prediction'] = {
                'date': pred_df.loc[worst_idx, 'date'].strftime('%Y-%m-%d'),
                'actual': float(pred_df.loc[worst_idx, 'actual']),
                'predicted': float(pred_df.loc[worst_idx, 'predicted']),
                'error': float(pred_df.loc[worst_idx, 'error'])
            }
        
        return predictions
    
    def _create_xgboost_charts(self, model: xgb.Booster, 
                              X_train: np.ndarray, X_test: np.ndarray,
                              y_train: np.ndarray, y_test: np.ndarray,
                              train_dates: pd.DatetimeIndex, test_dates: pd.DatetimeIndex,
                              importance: Dict[str, Any], symbol: str,
                              feature_names: List[str]) -> Dict[str, go.Figure]:
        """Create XGBoost visualization charts."""
        charts = {}
        
        # 1. Actual vs Predicted with Error Analysis
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.4, 0.3, 0.3],
            subplot_titles=(
                f'{symbol}: XGBoost Predictions vs Actual',
                'Prediction Error Over Time',
                'Error Distribution'
            )
        )
        
        # Make predictions
        dtrain = xgb.DMatrix(X_train)
        dtest = xgb.DMatrix(X_test)
        y_pred_train = model.predict(dtrain)
        y_pred_test = model.predict(dtest)
        
        # Combine for plotting
        all_dates = list(train_dates) + list(test_dates)
        all_actual = np.concatenate([y_train, y_test])
        all_predicted = np.concatenate([y_pred_train, y_pred_test])
        
        # Actual vs Predicted
        fig.add_trace(
            go.Scatter(
                x=all_dates,
                y=all_actual * 100,
                mode='lines',
                name='Actual',
                line=dict(color=self.color_palette['primary'], width=2),
                hovertemplate='Date: %{x}<br>Actual: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=all_dates,
                y=all_predicted * 100,
                mode='lines',
                name='XGBoost Predicted',
                line=dict(color=self.color_palette['success'], width=2, dash='dash'),
                hovertemplate='Date: %{x}<br>Predicted: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Error over time
        error_test = y_test - y_pred_test
        
        fig.add_trace(
            go.Scatter(
                x=test_dates,
                y=error_test * 100,
                mode='lines+markers',
                name='Prediction Error',
                line=dict(color=self.color_palette['danger'], width=1.5),
                marker=dict(size=4),
                hovertemplate='Date: %{x}<br>Error: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add zero line
        fig.add_hline(
            y=0,
            line_width=1,
            line_color=self.color_palette['text'],
            row=2, col=1
        )
        
        # Error distribution histogram
        fig.add_trace(
            go.Histogram(
                x=error_test * 100,
                nbinsx=30,
                name='Error Distribution',
                marker_color=self.color_palette['secondary'],
                opacity=0.7,
                hovertemplate='Error: %{x:.2f}%<br>Count: %{y}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Add normal distribution overlay
        if len(error_test) > 10:
            mu, sigma = np.mean(error_test * 100), np.std(error_test * 100)
            x_norm = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
            y_norm = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*((x_norm - mu)/sigma)**2)
            y_norm = y_norm * len(error_test) * (x_norm[1] - x_norm[0])  # Scale to histogram
            
            fig.add_trace(
                go.Scatter(
                    x=x_norm,
                    y=y_norm,
                    mode='lines',
                    name='Normal Distribution',
                    line=dict(color=self.color_palette['text'], width=2, dash='dash'),
                    hovertemplate='Normal Distribution<extra></extra>'
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            height=900,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Prediction Error (%)", row=3, col=1)
        fig.update_yaxes(title_text="Absolute Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Error (%)", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=3, col=1)
        
        charts['predictions_comprehensive'] = fig
        
        # 2. Feature Importance
        if 'by_gain' in importance and importance['by_gain']:
            fig2 = self._create_feature_importance_chart(importance['by_gain'], symbol)
            charts['feature_importance'] = fig2
        
        # 3. Learning Curves
        fig3 = self._create_learning_curves_chart(model, symbol)
        charts['learning_curves'] = fig3
        
        # 4. Prediction Scatter Plot
        fig4 = go.Figure()
        
        fig4.add_trace(
            go.Scatter(
                x=y_test * 100,
                y=y_pred_test * 100,
                mode='markers',
                name='Test Predictions',
                marker=dict(
                    size=8,
                    color=np.abs(error_test * 100),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Absolute Error (%)")
                ),
                hovertemplate='Actual: %{x:.2f}%<br>Predicted: %{y:.2f}%<br>Error: %{marker.color:.2f}%<extra></extra>'
            )
        )
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred_test.min()) * 100
        max_val = max(y_test.max(), y_pred_test.max()) * 100
        
        fig4.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color=self.color_palette['text'], width=1, dash='dash')
            )
        )
        
        fig4.update_layout(
            title=f'{symbol}: XGBoost Prediction Scatter Plot',
            xaxis_title='Actual Absolute Return (%)',
            yaxis_title='Predicted Absolute Return (%)',
            height=500,
            showlegend=True,
            template='plotly_white'
        )
        
        charts['prediction_scatter'] = fig4
        
        return charts
    
    def _create_feature_importance_chart(self, importance_list: List[Dict], 
                                        symbol: str) -> go.Figure:
        """Create feature importance chart."""
        fig = go.Figure()
        
        # Extract data
        features = [item['feature'] for item in importance_list[:15]]  # Top 15
        importance_values = [item['importance_normalized'] for item in importance_list[:15]]
        
        # Color by importance
        colors = [self.color_palette['primary']] * len(features)
        if importance_values:
            max_importance = max(importance_values)
            for i, val in enumerate(importance_values):
                if val == max_importance:
                    colors[i] = self.color_palette['warning']
        
        fig.add_trace(
            go.Bar(
                x=importance_values,
                y=features,
                orientation='h',
                marker_color=colors,
                text=[f'{v:.1f}%' for v in importance_values],
                textposition='auto',
                hovertemplate='Feature: %{y}<br>Importance: %{x:.1f}%<extra></extra>'
            )
        )
        
        fig.update_layout(
            title=f'{symbol}: XGBoost Feature Importance (Gain)',
            height=500,
            showlegend=False,
            template='plotly_white',
            xaxis_title='Importance (%)',
            yaxis_title='Feature',
            yaxis=dict(categoryorder='total ascending')
        )
        
        return fig
    
    def _create_learning_curves_chart(self, model: xgb.Booster, symbol: str) -> go.Figure:
        """Create learning curves chart from XGBoost."""
        fig = go.Figure()
        
        # Try to get evaluation history
        try:
            # XGBoost stores evaluation results in evals_result()
            evals_result = model.evals_result()
            
            if evals_result and 'train' in evals_result and 'val' in evals_result:
                train_rmse = evals_result['train'][self.params['eval_metric']]
                val_rmse = evals_result['val'][self.params['eval_metric']]
                
                iterations = list(range(1, len(train_rmse) + 1))
                
                fig.add_trace(
                    go.Scatter(
                        x=iterations,
                        y=train_rmse,
                        mode='lines',
                        name='Training RMSE',
                        line=dict(color=self.color_palette['primary'], width=2),
                        hovertemplate='Iteration: %{x}<br>RMSE: %{y:.4f}<extra></extra>'
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=iterations,
                        y=val_rmse,
                        mode='lines',
                        name='Validation RMSE',
                        line=dict(color=self.color_palette['success'], width=2),
                        hovertemplate='Iteration: %{x}<br>RMSE: %{y:.4f}<extra></extra>'
                    )
                )
                
                # Add early stopping line if applicable
                if hasattr(model, 'best_iteration') and model.best_iteration < len(iterations):
                    fig.add_vline(
                        x=model.best_iteration,
                        line_width=2,
                        line_dash="dash",
                        line_color=self.color_palette['warning'],
                        annotation_text=f"Best Iteration: {model.best_iteration}",
                        annotation_position="top right"
                    )
        except:
            # If evaluation history not available, create a simple placeholder
            self.logger.warning(f"Could not extract learning curves for {symbol}")
        
        fig.update_layout(
            title=f'{symbol}: XGBoost Learning Curves',
            xaxis_title='Number of Boosting Rounds',
            yaxis_title='RMSE',
            height=400,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def _generate_xgboost_insights(self, model: xgb.Booster, 
                                  evaluation: Dict[str, Any],
                                  importance: Dict[str, Any], symbol: str) -> List[str]:
        """Generate insights from XGBoost analysis."""
        insights = []
        
        # Model performance
        rmse = evaluation.get('test_rmse', 0)
        r2 = evaluation.get('test_r2', 0)
        corr = evaluation.get('test_correlation', 0)
        
        insights.append(
            f"XGBoost prediction RMSE: {rmse*100:.2f}%, "
            f"R²: {r2:.3f}, Correlation: {corr:.3f}"
        )
        
        # Performance interpretation
        if r2 > 0.4:
            insights.append("Strong predictive power for next-day volatility.")
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
