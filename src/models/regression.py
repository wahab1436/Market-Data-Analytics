"""
Regression Models Module
Linear, Ridge, and Lasso regression for interpretable market analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class RegressionModels:
    """Regression models for interpretable market analysis."""
    
    def __init__(self, config: Dict[str, Any], logger):
        """Initialize regression modeler."""
        self.config = config
        self.logger = logger
        self.color_palette = config['dashboard']['color_palette']
        self.models_path = Path(config['paths']['models'])
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.artifacts_path = Path(config['paths']['artifacts'])
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # Model parameters
        self.alpha_values = config['models']['regression']['alpha']
        self.test_size = config['models']['validation']['test_size']
        self.train_test_split = pd.to_datetime(config['models']['validation']['train_test_split'])
    
    def train_and_evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Train and evaluate regression models."""
        results = {}
        
        by_symbol = data['by_symbol']
        
        for symbol, df in by_symbol.items():
            self.logger.info(f"Training regression models for {symbol}")
            
            symbol_results = {
                'charts': {},
                'insights': [],
                'metrics': {},
                'models': {},
                'predictions': {}
            }
            
            # Prepare data for regression
            X, y, feature_names = self._prepare_regression_data(df, symbol)
            
            if X is None or len(X) < 20:
                self.logger.warning(f"Insufficient data for {symbol}")
                continue
            
            # Split data chronologically
            X_train, X_test, y_train, y_test, train_dates, test_dates = self._chronological_split(
                X, y, df.index
            )
            
            # Train models
            models = self._train_models(X_train, y_train, symbol)
            symbol_results['models'] = models
            
            # Evaluate models
            evaluation = self._evaluate_models(models, X_test, y_test, symbol)
            symbol_results['metrics'].update(evaluation)
            
            # Make predictions
            predictions = self._make_predictions(models, X_test, y_test, test_dates, symbol)
            symbol_results['predictions'] = predictions
            
            # Create regression charts
            regression_charts = self._create_regression_charts(
                models, X_train, X_test, y_train, y_test, 
                train_dates, test_dates, symbol, feature_names
            )
            symbol_results['charts'].update(regression_charts)
            
            # Generate insights
            insights = self._generate_regression_insights(models, evaluation, symbol)
            symbol_results['insights'].extend(insights)
            
            # Feature importance analysis
            importance_results = self._analyze_feature_importance(models, feature_names, symbol)
            symbol_results.update(importance_results)
            
            # Save models (saved separately to keep results lightweight)
            self._save_models(models, symbol)
            
            # Save symbol results to artifacts using SHARED MERGE LOGIC
            self._save_symbol_results(symbol, symbol_results)
            
            results[symbol] = symbol_results
        
        # Cross-symbol regression comparison
        if len(by_symbol) > 1:
            cross_results = self._analyze_cross_symbol_regression(results)
            results['cross_symbol'] = cross_results
        
        return results
    
    def _prepare_regression_data(self, df: pd.DataFrame, symbol: str) -> Tuple:
        """Prepare data for regression models."""
        # Define features and target
        feature_categories = [
            'return_lag_1d', 'return_lag_5d', 'return_lag_10d',
            'volume_ratio_10d', 'volume_ratio_20d',
            'volatility_10d', 'volatility_20d',
            'ma_20d', 'ma_50d',
            'price_vs_ma_20d_pct', 'price_vs_ma_50d_pct',
            'rolling_high_20d', 'rolling_low_20d',
            'daily_range_pct', 'atr_14d'
        ]
        
        # Select available features
        available_features = [f for f in feature_categories if f in df.columns]
        
        if len(available_features) < 5:
            self.logger.warning(f"Insufficient features for {symbol}")
            return None, None, None
        
        # Target: next day's absolute return (volatility proxy)
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
        
        if len(X) < 20:
            return None, None, None
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save scaler
        scaler_path = self.models_path / f"{symbol}_regression_scaler.pkl"
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
        
        self.logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test, train_dates, test_dates
    
    def _train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                     symbol: str) -> Dict[str, Any]:
        """Train regression models."""
        models = {}
        
        # Linear Regression
        self.logger.info(f"Training Linear Regression for {symbol}")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        models['linear'] = {
            'model': lr_model,
            'coef': lr_model.coef_,
            'intercept': lr_model.intercept_
        }
        
        # Ridge Regression
        self.logger.info(f"Training Ridge Regression for {symbol}")
        ridge_model = Ridge(alpha=self.alpha_values[0], random_state=42)
        ridge_model.fit(X_train, y_train)
        models['ridge'] = {
            'model': ridge_model,
            'coef': ridge_model.coef_,
            'intercept': ridge_model.intercept_
        }
        
        # Lasso Regression
        self.logger.info(f"Training Lasso Regression for {symbol}")
        lasso_model = Lasso(alpha=self.alpha_values[1], random_state=42, max_iter=10000)
        lasso_model.fit(X_train, y_train)
        models['lasso'] = {
            'model': lasso_model,
            'coef': lasso_model.coef_,
            'intercept': lasso_model.intercept_
        }
        
        # Additional Ridge with different alpha
        self.logger.info(f"Training Ridge (alpha={self.alpha_values[2]}) for {symbol}")
        ridge2_model = Ridge(alpha=self.alpha_values[2], random_state=42)
        ridge2_model.fit(X_train, y_train)
        models['ridge_high_alpha'] = {
            'model': ridge2_model,
            'coef': ridge2_model.coef_,
            'intercept': ridge2_model.intercept_
        }
        
        return models
    
    def _evaluate_models(self, models: Dict[str, Any], X_test: np.ndarray, 
                        y_test: np.ndarray, symbol: str) -> Dict[str, Any]:
        """Evaluate regression models."""
        evaluation = {}
        
        for name, model_info in models.items():
            model = model_info['model']
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            evaluation[f'{name}_mse'] = float(mse)
            evaluation[f'{name}_rmse'] = float(rmse)
            evaluation[f'{name}_mae'] = float(mae)
            evaluation[f'{name}_r2'] = float(r2)
            
            # Direction accuracy (sign prediction)
            if len(y_test) > 0 and len(y_pred) > 0:
                # For volatility prediction, we care about magnitude
                # Calculate correlation between actual and predicted
                corr = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else 0
                evaluation[f'{name}_correlation'] = float(corr)
                
                # Percent within 1 standard deviation
                error = np.abs(y_test - y_pred)
                std_actual = np.std(y_test)
                within_1std = (error < std_actual).mean()
                evaluation[f'{name}_within_1std'] = float(within_1std)
        
        # Identify best model
        best_model_name = None
        best_rmse = float('inf')
        
        for name in ['linear', 'ridge', 'lasso', 'ridge_high_alpha']:
            rmse_key = f'{name}_rmse'
            if rmse_key in evaluation and evaluation[rmse_key] < best_rmse:
                best_rmse = evaluation[rmse_key]
                best_model_name = name
        
        if best_model_name:
            evaluation['best_model'] = best_model_name
            evaluation['best_rmse'] = best_rmse
        
        self.logger.info(f"Best model for {symbol}: {best_model_name} (RMSE: {best_rmse:.4f})")
        
        return evaluation
    
    def _make_predictions(self, models: Dict[str, Any], X_test: np.ndarray,
                         y_test: np.ndarray, test_dates: pd.DatetimeIndex,
                         symbol: str) -> Dict[str, Any]:
        """Generate predictions for visualization."""
        predictions = {}
        
        # Use best model for predictions
        best_model_name = None
        best_rmse = float('inf')
        
        for name in ['linear', 'ridge', 'lasso', 'ridge_high_alpha']:
            if f'{name}_rmse' in models:  # Check if evaluated
                rmse = models.get(f'{name}_rmse', float('inf'))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model_name = name
        
        if best_model_name and best_model_name in models:
            best_model_info = models[best_model_name]
            model = best_model_info['model']
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Create prediction DataFrame
            pred_df = pd.DataFrame({
                'date': test_dates,
                'actual': y_test,
                'predicted': y_pred,
                'error': y_test - y_pred,
                'abs_error': np.abs(y_test - y_pred)
            })
            
            # Rolling performance metrics
            window = min(20, len(pred_df))
            if window > 0:
                pred_df['rolling_mae'] = pred_df['abs_error'].rolling(window=window).mean()
                pred_df['rolling_correlation'] = pred_df['actual'].rolling(window=window).corr(pred_df['predicted'])
            
            predictions['best_model'] = best_model_name
            predictions['data'] = pred_df.to_dict('records')
            predictions['summary'] = {
                'mean_actual': float(np.mean(y_test)),
                'mean_predicted': float(np.mean(y_pred)),
                'std_actual': float(np.std(y_test)),
                'std_predicted': float(np.std(y_pred)),
                'correlation': float(np.corrcoef(y_test, y_pred)[0, 1]) if len(y_test) > 1 else 0
            }
        
        return predictions
    
    def _create_regression_charts(self, models: Dict[str, Any], 
                                X_train: np.ndarray, X_test: np.ndarray,
                                y_train: np.ndarray, y_test: np.ndarray,
                                train_dates: pd.DatetimeIndex, test_dates: pd.DatetimeIndex,
                                symbol: str, feature_names: List[str]) -> Dict[str, go.Figure]:
        """Create regression visualization charts."""
        charts = {}
        
        # 1. Actual vs Predicted Comparison
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4],
            subplot_titles=(
                f'{symbol}: Actual vs Predicted Next-Day Absolute Returns',
                'Prediction Error Over Time'
            )
        )
        
        # Get predictions from best model
        best_model_name = None
        best_model = None
        
        for name in ['linear', 'ridge', 'lasso', 'ridge_high_alpha']:
            if name in models:
                best_model_name = name
                best_model = models[name]['model']
                break
        
        if best_model:
            # Predictions
            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)
            
            # Combine train and test for plotting
            all_dates = list(train_dates) + list(test_dates)
            all_actual = np.concatenate([y_train, y_test])
            all_predicted = np.concatenate([y_pred_train, y_pred_test])
            
            # Scatter plot of actual vs predicted
            fig.add_trace(
                go.Scatter(
                    x=all_actual * 100,  # Convert to percentage
                    y=all_predicted * 100,
                    mode='markers',
                    name=f'{best_model_name.capitalize()} Predictions',
                    marker=dict(
                        size=8,
                        color=np.where(np.arange(len(all_actual)) < len(y_train),
                                     self.color_palette['primary'],  # Train points
                                     self.color_palette['secondary']),  # Test points
                        opacity=0.7,
                        symbol=np.where(np.arange(len(all_actual)) < len(y_train),
                                      'circle', 'circle-open')
                    ),
                    text=[f"Date: {d.strftime('%Y-%m-%d')}<br>"
                          f"Actual: {a*100:.2f}%<br>Predicted: {p*100:.2f}%"
                          for d, a, p in zip(all_dates, all_actual, all_predicted)],
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Perfect prediction line
            min_val = min(all_actual.min(), all_predicted.min()) * 100
            max_val = max(all_actual.max(), all_predicted.max()) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color=self.color_palette['text'], width=1, dash='dash'),
                    showlegend=True
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
            
            # Add rolling mean of error
            window = min(20, len(error_test))
            if window > 0:
                rolling_error = pd.Series(error_test).rolling(window=window).mean()
                fig.add_trace(
                    go.Scatter(
                        x=test_dates,
                        y=rolling_error * 100,
                        mode='lines',
                        name=f'{window}-day Rolling Mean Error',
                        line=dict(color=self.color_palette['warning'], width=2, dash='dash'),
                        hovertemplate='Rolling Error: %{y:.2f}%<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        fig.update_layout(
            height=700,
            showlegend=True,
            hovermode='closest',
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Actual Absolute Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Absolute Return (%)", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Prediction Error (%)", row=2, col=1)
        
        charts['actual_vs_predicted'] = fig
        
        # 2. Model Comparison Bar Chart
        fig2 = go.Figure()
        
        model_names = []
        rmse_values = []
        r2_values = []
        
        for name in ['linear', 'ridge', 'lasso', 'ridge_high_alpha']:
            if name in models:
                model_names.append(name.capitalize())
                rmse_key = f'{name}_rmse'
                r2_key = f'{name}_r2'
                
                # Get metrics from model evaluation
                if hasattr(models[name]['model'], 'score'):
                    rmse_values.append(models[name].get(rmse_key, 0) * 100)  # Convert to percentage
                    r2_values.append(models[name].get(r2_key, 0))
        
        if rmse_values:
            # RMSE comparison
            fig2.add_trace(
                go.Bar(
                    x=model_names,
                    y=rmse_values,
                    name='RMSE (%)',
                    marker_color=self.color_palette['primary'],
                    text=[f'{v:.3f}%' for v in rmse_values],
                    textposition='auto',
                    hovertemplate='Model: %{x}<br>RMSE: %{y:.3f}%<extra></extra>'
                )
            )
            
            # R² comparison (secondary axis)
            fig2.add_trace(
                go.Scatter(
                    x=model_names,
                    y=r2_values,
                    name='R² Score',
                    mode='lines+markers',
                    yaxis='y2',
                    line=dict(color=self.color_palette['success'], width=3),
                    marker=dict(size=10),
                    text=[f'{v:.3f}' for v in r2_values],
                    hovertemplate='Model: %{x}<br>R²: %{y:.3f}<extra></extra>'
                )
            )
            
            fig2.update_layout(
                title=f'{symbol}: Model Performance Comparison',
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
            
            charts['model_comparison'] = fig2
        
        # 3. Coefficient Analysis (if we have feature names)
        if feature_names and best_model_name and best_model_name in models:
            fig3 = self._create_coefficient_chart(
                models[best_model_name], feature_names, symbol, best_model_name
            )
            charts['coefficient_analysis'] = fig3
        
        return charts
    
    def _create_coefficient_chart(self, model_info: Dict[str, Any], 
                                 feature_names: List[str], symbol: str, 
                                 model_name: str) -> go.Figure:
        """Create coefficient importance chart."""
        fig = go.Figure()
        
        coefficients = model_info['coef']
        
        if len(coefficients) != len(feature_names):
            return fig
        
        # Sort by absolute coefficient value
        sorted_indices = np.argsort(np.abs(coefficients))[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices[:15]]  # Top 15
        sorted_coeffs = [coefficients[i] for i in sorted_indices[:15]]
        
        # Color by sign
        colors = [self.color_palette['success'] if c >= 0 
                 else self.color_palette['danger'] for c in sorted_coeffs]
        
        fig.add_trace(
            go.Bar(
                x=sorted_coeffs,
                y=sorted_features,
                orientation='h',
                marker_color=colors,
                text=[f'{c:.4f}' for c in sorted_coeffs],
                textposition='auto',
                hovertemplate='Feature: %{y}<br>Coefficient: %{x:.4f}<extra></extra>'
            )
        )
        
        # Add vertical line at 0
        fig.add_vline(
            x=0,
            line_width=1,
            line_color=self.color_palette['text']
        )
        
        fig.update_layout(
            title=f'{symbol}: {model_name.capitalize()} Feature Coefficients',
            height=500,
            showlegend=False,
            template='plotly_white',
            xaxis_title='Coefficient Value',
            yaxis_title='Feature',
            yaxis=dict(categoryorder='total ascending')
        )
        
        return fig
    
    def _generate_regression_insights(self, models: Dict[str, Any], 
                                     evaluation: Dict[str, Any], 
                                     symbol: str) -> List[str]:
        """Generate insights from regression analysis."""
        insights = []
        
        if not evaluation:
            return insights
        
        # Best model insight
        best_model = evaluation.get('best_model')
        best_rmse = evaluation.get('best_rmse', 0)
        
        if best_model:
            insights.append(
                f"Best regression model: {best_model.capitalize()} "
                f"(RMSE: {best_rmse*100:.2f}%, R²: {evaluation.get(f'{best_model}_r2', 0):.3f})"
            )
        
        # Model performance comparison
        model_performance = []
        for name in ['linear', 'ridge', 'lasso', 'ridge_high_alpha']:
            r2_key = f'{name}_r2'
            if r2_key in evaluation:
                r2 = evaluation[r2_key]
                model_performance.append((name, r2))
        
        if model_performance:
            # Check if regularization helps
            linear_r2 = evaluation.get('linear_r2', 0)
            ridge_r2 = evaluation.get('ridge_r2', 0)
            lasso_r2 = evaluation.get('lasso_r2', 0)
            
            if ridge_r2 > linear_r2 + 0.02:
                insights.append("Ridge regression outperforms linear, suggesting some multicollinearity.")
            elif lasso_r2 > linear_r2 + 0.02:
                insights.append("Lasso regression outperforms linear, indicating sparse feature importance.")
            
            # Check R² quality
            best_r2 = max([p[1] for p in model_performance])
            if best_r2 > 0.3:
                insights.append(f"Good explanatory power (R²: {best_r2:.3f}) for volatility prediction.")
            elif best_r2 < 0.1:
                insights.append(f"Limited explanatory power (R²: {best_r2:.3f}), suggesting hard-to-predict volatility.")
        
        # Error analysis
        for name in ['linear', 'ridge', 'lasso']:
            within_1std_key = f'{name}_within_1std'
            if within_1std_key in evaluation:
                within_1std = evaluation[within_1std_key]
                if within_1std > 0.7:
                    insights.append(f"{name.capitalize()} predictions frequently within 1 standard deviation.")
                elif within_1std < 0.5:
                    insights.append(f"{name.capitalize()} predictions often outside 1 standard deviation.")
                break
        
        return insights
    
    def _analyze_feature_importance(self, models: Dict[str, Any], 
                                   feature_names: List[str], symbol: str) -> Dict[str, Any]:
        """Analyze feature importance across models."""
        results = {}
        
        importance_data = {}
        
        for name, model_info in models.items():
            if 'coef' in model_info:
                coefficients = model_info['coef']
                
                # Calculate absolute importance
                abs_coeff = np.abs(coefficients)
                if np.sum(abs_coeff) > 0:
                    importance = abs_coeff / np.sum(abs_coeff)
                else:
                    importance = np.zeros_like(abs_coeff)
                
                importance_data[name] = {
                    'coefficients': coefficients.tolist() if hasattr(coefficients, 'tolist') else list(coefficients),
                    'importance': importance.tolist() if hasattr(importance, 'tolist') else list(importance)
                }
        
        # Calculate consensus importance
        if importance_data:
            n_models = len(importance_data)
            consensus_importance = np.zeros(len(feature_names))
            
            for model_data in importance_data.values():
                consensus_importance += np.array(model_data['importance'])
            
            consensus_importance /= n_models
            
            # Get top features
            top_indices = np.argsort(consensus_importance)[::-1][:10]
            top_features = [feature_names[i] for i in top_indices]
            top_importance = consensus_importance[top_indices]
            
            results['feature_importance'] = {
                'consensus': {
                    'features': top_features,
                    'importance': top_importance.tolist()
                },
                'by_model': importance_data
            }
        
        return results
    
    def _save_models(self, models: Dict[str, Any], symbol: str):
        """Save trained models to disk."""
        for name, model_info in models.items():
            model_path = self.models_path / f"{symbol}_regression_{name}.pkl"
            joblib.dump(model_info['model'], model_path)
        
        self.logger.debug(f"Saved regression models for {symbol}")
    
    def _analyze_cross_symbol_regression(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze regression results across symbols."""
        cross_results = {}
        
        symbols = [s for s in results.keys() if s != 'cross_symbol']
        
        if len(symbols) < 2:
            return cross_results
        
        # Collect performance metrics
        performance_data = []
        
        for symbol in symbols:
            symbol_data = results[symbol]
            metrics = symbol_data.get('metrics', {})
            
            best_model = metrics.get('best_model')
            best_rmse = metrics.get('best_rmse', 0)
            best_r2 = metrics.get(f'{best_model}_r2', 0) if best_model else 0
            
            performance_data.append({
                'symbol': symbol,
                'best_model': best_model,
                'rmse': best_rmse,
                'r2': best_r2
            })
        
        # Create comparison chart
        fig = go.Figure()
        
        symbols_list = [p['symbol'] for p in performance_data]
        rmse_values = [p['rmse'] * 100 for p in performance_data]  # Convert to percentage
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
            title='Cross-Symbol Regression Performance Comparison',
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
        
        # Find best and worst performing symbols
        if performance_data:
            best_by_rmse = min(performance_data, key=lambda x: x['rmse'])
            worst_by_rmse = max(performance_data, key=lambda x: x['rmse'])
            
            insights.append(
                f"Best volatility prediction: {best_by_rmse['symbol']} "
                f"(RMSE: {best_by_rmse['rmse']*100:.2f}%)"
            )
            insights.append(
                f"Most challenging prediction: {worst_by_rmse['symbol']} "
                f"(RMSE: {worst_by_rmse['rmse']*100:.2f}%)"
            )
            
            # Check model consistency
            models_used = [p['best_model'] for p in performance_data if p['best_model']]
            if len(set(models_used)) == 1:
                insights.append(f"Consistent best model across symbols: {models_used[0].capitalize()}")
            else:
                insights.append(f"Different best models across symbols: {', '.join(set(models_used))}")
        
        cross_results['insights'] = insights
        
        return cross_results

    def _save_symbol_results(self, symbol: str, results: Dict[str, Any]) -> None:
        """
        FIXED: Merges regression results into the unified _analysis_results.pkl file.
        Uses Load-Update-Save pattern to preserve data from other modules.
        """
        try:
            self.artifacts_path.mkdir(parents=True, exist_ok=True)
            output_file = self.artifacts_path / "_analysis_results.pkl"
            
            # 1. Prepare data (filter out heavy objects)
            save_results = {k: v for k, v in results.items() 
                           if k not in ['models', 'charts', 'fig', 'plots']}
            
            # 2. Load existing
            final_data = {}
            if output_file.exists():
                try:
                    final_data = joblib.load(output_file)
                    if not isinstance(final_data, dict):
                        final_data = {}
                except Exception as e:
                    self.logger.warning(f"Could not load existing results: {e}")
                    final_data = {}
            
            # 3. Ensure 'regression' key exists in unified structure
            if 'regression' not in final_data:
                final_data['regression'] = {}
            
            # 4. Update this symbol's regression data (nested under regression key)
            final_data['regression'][symbol] = save_results
            
            # 5. Save back to unified file
            joblib.dump(final_data, output_file)
            self.logger.info(f"Merged regression results for {symbol} into {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving regression results for {symbol}: {str(e)}")