"""
Utilities and Helper Functions
Shared functionality for the Market Insight Platform
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import json
import hashlib
import warnings
import re


class MarketDataHelpers:
    """Helper functions for market data processing and analysis."""
    
    @staticmethod
    def validate_date_range(start_date: str, end_date: str) -> Tuple[bool, str]:
        """Validate date range parameters."""
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            if start > end:
                return False, "Start date must be before end date"
            
            if start > pd.Timestamp.now():
                return False, "Start date cannot be in the future"
            
            if end > pd.Timestamp.now() + timedelta(days=1):
                return False, "End date cannot be more than 1 day in the future"
            
            # Check if range is too short
            if (end - start).days < 5:
                return False, "Date range must be at least 5 days"
            
            # Check if range is too long (for free tier)
            if (end - start).days > 365 * 2:
                return False, "Date range cannot exceed 2 years for free tier"
            
            return True, "Date range is valid"
            
        except Exception as e:
            return False, f"Invalid date format: {str(e)}"
    
    @staticmethod
    def calculate_trading_days(start_date: datetime, end_date: datetime) -> int:
        """Estimate number of trading days between dates."""
        # Simple estimation: 252 trading days per year
        days_diff = (end_date - start_date).days
        trading_days = int(days_diff * 252 / 365)
        return max(1, trading_days)
    
    @staticmethod
    def clean_symbol(symbol: str) -> str:
        """Clean and validate symbol string."""
        if not symbol or not isinstance(symbol, str):
            return ""
        
        # Remove whitespace and convert to uppercase
        cleaned = symbol.strip().upper()
        
        # Remove non-alphanumeric characters except dots
        cleaned = re.sub(r'[^A-Z0-9.]', '', cleaned)
        
        return cleaned
    
    @staticmethod
    def validate_symbol(symbol: str) -> Tuple[bool, str]:
        """Validate stock symbol format."""
        if not symbol:
            return False, "Symbol cannot be empty"
        
        if len(symbol) > 10:
            return False, "Symbol too long (max 10 characters)"
        
        # Basic pattern for stock symbols
        pattern = r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$'
        if not re.match(pattern, symbol):
            return False, "Invalid symbol format"
        
        return True, "Symbol is valid"
    
    @staticmethod
    def calculate_performance_metrics(prices: pd.Series) -> Dict[str, float]:
        """Calculate basic performance metrics from price series."""
        if len(prices) < 2:
            return {}
        
        returns = prices.pct_change().dropna()
        
        metrics = {
            'total_return': float((prices.iloc[-1] / prices.iloc[0] - 1) * 100),
            'annualized_return': float(((1 + returns.mean()) ** 252 - 1) * 100),
            'annualized_volatility': float(returns.std() * np.sqrt(252) * 100),
            'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
            'max_drawdown': float(MarketDataHelpers.calculate_max_drawdown(prices)),
            'calmar_ratio': 0.0
        }
        
        if metrics['max_drawdown'] != 0:
            metrics['calmar_ratio'] = float(metrics['annualized_return'] / abs(metrics['max_drawdown']))
        
        # Sortino ratio (only downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        if downside_std > 0:
            metrics['sortino_ratio'] = float(returns.mean() / downside_std * np.sqrt(252))
        else:
            metrics['sortino_ratio'] = 0.0
        
        return metrics
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """Calculate maximum drawdown in percentage."""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return float(drawdown.min() * 100)
    
    @staticmethod
    def detect_outliers(series: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
        """Detect outliers in a series using specified method."""
        if method == 'iqr':
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            return (series < lower_bound) | (series > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            return z_scores > threshold
        
        elif method == 'mad':
            # Median Absolute Deviation
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z_scores = 0.6745 * (series - median) / mad
            return np.abs(modified_z_scores) > threshold
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    @staticmethod
    def calculate_rolling_correlation(series1: pd.Series, series2: pd.Series, 
                                     window: int = 20) -> pd.Series:
        """Calculate rolling correlation between two series."""
        aligned = pd.concat([series1, series2], axis=1).dropna()
        if len(aligned) < window:
            return pd.Series(index=aligned.index, dtype=float)
        
        return aligned.iloc[:, 0].rolling(window=window).corr(aligned.iloc[:, 1])
    
    @staticmethod
    def create_lagged_features(df: pd.DataFrame, columns: List[str], 
                              lags: List[int]) -> pd.DataFrame:
        """Create lagged versions of specified columns."""
        df_lagged = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for lag in lags:
                df_lagged[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df_lagged
    
    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate common technical indicators."""
        df_tech = df.copy()
        
        if 'close' not in df.columns:
            return df_tech
        
        # Moving averages
        for period in [10, 20, 50, 200]:
            df_tech[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
        
        # Exponential moving averages
        for period in [12, 26]:
            df_tech[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # MACD
        if 'EMA_12' in df_tech.columns and 'EMA_26' in df_tech.columns:
            df_tech['MACD'] = df_tech['EMA_12'] - df_tech['EMA_26']
            df_tech['MACD_Signal'] = df_tech['MACD'].ewm(span=9, adjust=False).mean()
            df_tech['MACD_Histogram'] = df_tech['MACD'] - df_tech['MACD_Signal']
        
        # RSI
        if 'close' in df.columns:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_tech['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        if 'close' in df.columns:
            df_tech['BB_Middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df_tech['BB_Upper'] = df_tech['BB_Middle'] + (bb_std * 2)
            df_tech['BB_Lower'] = df_tech['BB_Middle'] - (bb_std * 2)
            df_tech['BB_Width'] = (df_tech['BB_Upper'] - df_tech['BB_Lower']) / df_tech['BB_Middle']
        
        return df_tech
    
    @staticmethod
    def calculate_returns_statistics(returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive statistics for return series."""
        if len(returns) < 2:
            return {}
        
        stats = {
            'mean': float(returns.mean()),
            'std': float(returns.std()),
            'skewness': float(returns.skew()),
            'kurtosis': float(returns.kurtosis()),
            'min': float(returns.min()),
            'max': float(returns.max()),
            'median': float(returns.median()),
            'mad': float((returns - returns.mean()).abs().mean()),  # Mean Absolute Deviation
            'positive_ratio': float((returns > 0).mean()),
            'negative_ratio': float((returns < 0).mean()),
            'zero_ratio': float((returns == 0).mean())
        }
        
        # VaR (Value at Risk) at different confidence levels
        for confidence in [90, 95, 99]:
            var = np.percentile(returns, 100 - confidence)
            stats[f'var_{confidence}'] = float(var)
        
        # Expected Shortfall (Conditional VaR)
        for confidence in [90, 95, 99]:
            var_level = 100 - confidence
            var = np.percentile(returns, var_level)
            es = returns[returns <= var].mean()
            stats[f'es_{confidence}'] = float(es)
        
        # Autocorrelation
        for lag in [1, 5, 10]:
            if len(returns) > lag:
                autocorr = returns.autocorr(lag=lag)
                stats[f'autocorr_lag_{lag}'] = float(autocorr)
        
        return stats


class DataStorageHelpers:
    """Helper functions for data storage and retrieval."""
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """Ensure directory exists, create if it doesn't."""
        path_obj = Path(path) if isinstance(path, str) else path
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    
    @staticmethod
    def save_dataframe(df: pd.DataFrame, path: Union[str, Path], 
                      format: str = 'parquet') -> bool:
        """Save DataFrame to disk in specified format."""
        try:
            path_obj = Path(path) if isinstance(path, str) else path
            
            if format.lower() == 'parquet':
                df.to_parquet(path_obj, index=False, compression='snappy')
            elif format.lower() == 'csv':
                df.to_csv(path_obj, index=False)
            elif format.lower() == 'json':
                df.to_json(path_obj, orient='records', indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to save DataFrame: {e}")
            return False
    
    @staticmethod
    def load_dataframe(path: Union[str, Path], 
                      format: str = None) -> Optional[pd.DataFrame]:
        """Load DataFrame from disk, auto-detect format from extension."""
        try:
            path_obj = Path(path) if isinstance(path, str) else path
            
            if not path_obj.exists():
                return None
            
            # Auto-detect format from extension
            if format is None:
                suffix = path_obj.suffix.lower()
                if suffix == '.parquet':
                    format = 'parquet'
                elif suffix == '.csv':
                    format = 'csv'
                elif suffix == '.json':
                    format = 'json'
                else:
                    raise ValueError(f"Cannot auto-detect format from extension: {suffix}")
            
            if format.lower() == 'parquet':
                df = pd.read_parquet(path_obj)
            elif format.lower() == 'csv':
                df = pd.read_csv(path_obj)
            elif format.lower() == 'json':
                df = pd.read_json(path_obj, orient='records')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return df
            
        except Exception as e:
            warnings.warn(f"Failed to load DataFrame: {e}")
            return None
    
    @staticmethod
    def generate_cache_key(symbol: str, start_date: str, end_date: str, 
                          data_type: str = 'ohlcv') -> str:
        """Generate a unique cache key for data."""
        key_string = f"{symbol}_{start_date}_{end_date}_{data_type}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    @staticmethod
    def is_cache_valid(cache_path: Union[str, Path], 
                      max_age_hours: int = 24) -> bool:
        """Check if cache file is still valid based on age."""
        cache_file = Path(cache_path) if isinstance(cache_path, str) else cache_path
        
        if not cache_file.exists():
            return False
        
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age.total_seconds() < (max_age_hours * 3600)
    
    @staticmethod
    def clean_old_cache(cache_dir: Union[str, Path], max_age_hours: int = 24):
        """Remove old cache files."""
        cache_dir_path = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
        
        if not cache_dir_path.exists():
            return
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        for cache_file in cache_dir_path.glob('*'):
            if cache_file.is_file():
                file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_mtime < cutoff_time:
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        warnings.warn(f"Failed to delete old cache file {cache_file}: {e}")


class StatisticalHelpers:
    """Statistical helper functions."""
    
    @staticmethod
    def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for data."""
        if len(data) < 2:
            return (0, 0)
        
        mean = np.mean(data)
        std_err = np.std(data) / np.sqrt(len(data))
        
        # Z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        margin = z_score * std_err
        return (float(mean - margin), float(mean + margin))
    
    @staticmethod
    def calculate_bootstrap_statistic(data: np.ndarray, statistic_func, 
                                     n_bootstrap: int = 1000, 
                                     confidence: float = 0.95) -> Dict[str, float]:
        """Calculate bootstrap confidence intervals for a statistic."""
        if len(data) < 10:
            return {}
        
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            sample = np.random.choice(data, size=len(data), replace=True)
            stat = statistic_func(sample)
            bootstrap_stats.append(stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate confidence intervals
        lower_percentile = (1 - confidence) / 2 * 100
        upper_percentile = (1 + confidence) / 2 * 100
        
        return {
            'mean': float(np.mean(bootstrap_stats)),
            'std': float(np.std(bootstrap_stats)),
            'ci_lower': float(np.percentile(bootstrap_stats, lower_percentile)),
            'ci_upper': float(np.percentile(bootstrap_stats, upper_percentile)),
            'median': float(np.median(bootstrap_stats))
        }
    
    @staticmethod
    def test_normality(data: np.ndarray) -> Dict[str, float]:
        """Test if data follows normal distribution."""
        from scipy import stats
        
        results = {}
        
        # Shapiro-Wilk test
        if len(data) >= 3 and len(data) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(data)
            results['shapiro_stat'] = float(shapiro_stat)
            results['shapiro_p'] = float(shapiro_p)
            results['shapiro_normal'] = shapiro_p > 0.05
        
        # Anderson-Darling test
        anderson_result = stats.anderson(data, dist='norm')
        results['anderson_stat'] = float(anderson_result.statistic)
        results['anderson_critical'] = anderson_result.critical_values.tolist()
        results['anderson_significant'] = anderson_result.statistic > anderson_result.critical_values[2]
        
        # Jarque-Bera test
        jb_stat, jb_p = stats.jarque_bera(data)
        results['jarque_bera_stat'] = float(jb_stat)
        results['jarque_bera_p'] = float(jb_p)
        results['jarque_bera_normal'] = jb_p > 0.05
        
        return results
    
    @staticmethod
    def calculate_autocorrelation_plot(data: pd.Series, max_lags: int = 40) -> Dict[str, List]:
        """Calculate autocorrelation and partial autocorrelation."""
        from statsmodels.tsa.stattools import acf, pacf
        
        if len(data) < max_lags * 2:
            max_lags = len(data) // 2
        
        acf_values = acf(data, nlags=max_lags, fft=True)
        pacf_values = pacf(data, nlags=max_lags)
        
        # Calculate confidence intervals
        conf_int = 1.96 / np.sqrt(len(data))
        
        return {
            'lags': list(range(max_lags + 1)),
            'acf': acf_values.tolist(),
            'pacf': pacf_values.tolist(),
            'conf_int_upper': conf_int,
            'conf_int_lower': -conf_int
        }


class TimeSeriesHelpers:
    """Time series specific helper functions."""
    
    @staticmethod
    def resample_time_series(df: pd.DataFrame, freq: str = 'B') -> pd.DataFrame:
        """Resample time series to business day frequency."""
        if df.index.isna().any():
            df = df.dropna(subset=[df.index.name or 'index'])
        
        df_resampled = df.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return df_resampled
    
    @staticmethod
    def fill_missing_dates(df: pd.DataFrame, freq: str = 'B') -> pd.DataFrame:
        """Fill missing dates in time series."""
        if df.empty:
            return df
        
        # Create complete date range
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
        
        # Reindex to fill missing dates
        df_filled = df.reindex(full_range)
        
        # Forward fill price data, zero fill volume
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].ffill().bfill()
        
        if 'volume' in df_filled.columns:
            df_filled['volume'] = df_filled['volume'].fillna(0)
        
        return df_filled
    
    @staticmethod
    def detect_structural_breaks(prices: pd.Series, method: str = 'cusum') -> List[datetime]:
        """Detect structural breaks in time series."""
        from scipy import stats
        
        breaks = []
        
        if method == 'cusum':
            # CUSUM test for structural breaks
            cumulative_sum = (prices - prices.mean()).cumsum()
            cusum_abs = cumulative_sum.abs()
            
            # Find points where CUSUM exceeds threshold
            threshold = cusum_abs.std() * 2
            break_indices = np.where(cusum_abs > threshold)[0]
            
            for idx in break_indices:
                if idx > 0 and idx < len(prices) - 1:
                    breaks.append(prices.index[idx])
        
        elif method == 'chow':
            # Chow test for structural breaks (simplified)
            n = len(prices)
            test_points = range(n // 4, 3 * n // 4, n // 20)
            
            for t in test_points:
                if t < 10 or n - t < 10:
                    continue
                
                # Split data
                y1 = prices.iloc[:t]
                y2 = prices.iloc[t:]
                
                # Calculate statistics
                var_total = prices.var()
                var1 = y1.var()
                var2 = y2.var()
                
                # Simple chow-like test
                if var1 > 2 * var_total or var2 > 2 * var_total:
                    breaks.append(prices.index[t])
        
        return breaks
    
    @staticmethod
    def calculate_rolling_regression(x: pd.Series, y: pd.Series, 
                                    window: int = 20) -> Dict[str, pd.Series]:
        """Calculate rolling linear regression parameters."""
        from scipy import stats
        
        results = {
            'slope': pd.Series(index=x.index, dtype=float),
            'intercept': pd.Series(index=x.index, dtype=float),
            'r_squared': pd.Series(index=x.index, dtype=float),
            'p_value': pd.Series(index=x.index, dtype=float)
        }
        
        for i in range(window, len(x)):
            x_window = x.iloc[i-window:i]
            y_window = y.iloc[i-window:i]
            
            if len(x_window) < 2 or len(y_window) < 2:
                continue
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_window, y_window)
            
            results['slope'].iloc[i] = slope
            results['intercept'].iloc[i] = intercept
            results['r_squared'].iloc[i] = r_value ** 2
            results['p_value'].iloc[i] = p_value
        
        return results


class ValidationHelpers:
    """Data validation helper functions."""
    
    @staticmethod
    def validate_ohlcv_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate OHLCV DataFrame for required fields and data quality."""
        errors = []
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            return False, errors
        
        # Check for NaN values
        nan_counts = df[required_columns].isna().sum()
        if nan_counts.any():
            errors.append(f"NaN values found: {nan_counts.to_dict()}")
        
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        negative_prices = (df[price_columns] <= 0).any().any()
        if negative_prices:
            errors.append("Found non-positive prices")
        
        # Check for negative volume
        negative_volume = (df['volume'] < 0).any()
        if negative_volume:
            errors.append("Found negative volume")
        
        # Check high-low consistency
        high_low_invalid = (df['high'] < df['low']).any()
        if high_low_invalid:
            errors.append("High < Low inconsistency found")
        
        # Check open/close within high/low range
        open_range_invalid = ((df['open'] < df['low']) | (df['open'] > df['high'])).any()
        close_range_invalid = ((df['close'] < df['low']) | (df['close'] > df['high'])).any()
        
        if open_range_invalid:
            errors.append("Open price outside high/low range")
        if close_range_invalid:
            errors.append("Close price outside high/low range")
        
        # Check for large gaps (potential errors)
        if 'close' in df.columns and len(df) > 1:
            returns = df['close'].pct_change().dropna()
            extreme_returns = returns.abs() > 0.25  # >25% daily move
            if extreme_returns.any():
                errors.append(f"Found {extreme_returns.sum()} extreme returns (>25%)")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def check_data_leakage(train_dates: pd.DatetimeIndex, 
                          test_dates: pd.DatetimeIndex) -> Tuple[bool, str]:
        """Check for data leakage between train and test sets."""
        if train_dates.empty or test_dates.empty:
            return False, "Empty date ranges"
        
        # Check if test dates are after train dates
        if test_dates.min() < train_dates.max():
            return False, "Test data contains dates before training data ends"
        
        # Check for overlap
        train_set = set(train_dates)
        test_set = set(test_dates)
        overlap = train_set.intersection(test_set)
        
        if overlap:
            return False, f"Found {len(overlap)} overlapping dates between train and test"
        
        return True, "No data leakage detected"
    
    @staticmethod
    def validate_feature_matrix(X: np.ndarray, y: np.ndarray = None) -> Tuple[bool, List[str]]:
        """Validate feature matrix for ML models."""
        errors = []
        
        # Check for NaN values
        if np.isnan(X).any():
            errors.append("Feature matrix contains NaN values")
        
        # Check for infinite values
        if not np.isfinite(X).all():
            errors.append("Feature matrix contains infinite values")
        
        # Check for constant features
        if X.shape[1] > 0:
            feature_stds = np.std(X, axis=0)
            constant_features = np.where(feature_stds == 0)[0]
            if len(constant_features) > 0:
                errors.append(f"Found {len(constant_features)} constant features")
        
        # Check shape consistency with labels if provided
        if y is not None:
            if len(X) != len(y):
                errors.append(f"X and y have different lengths: {len(X)} vs {len(y)}")
            
            if np.isnan(y).any():
                errors.append("Target variable contains NaN values")
        
        return len(errors) == 0, errors


class FormattingHelpers:
    """Formatting and display helper functions."""
    
    @staticmethod
    def format_currency(value: float, decimals: int = 2) -> str:
        """Format number as currency."""
        if pd.isna(value):
            return "N/A"
        
        if abs(value) >= 1_000_000_000:
            return f"${value/1_000_000_000:.{decimals}f}B"
        elif abs(value) >= 1_000_000:
            return f"${value/1_000_000:.{decimals}f}M"
        elif abs(value) >= 1_000:
            return f"${value/1_000:.{decimals}f}K"
        else:
            return f"${value:.{decimals}f}"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 2) -> str:
        """Format number as percentage."""
        if pd.isna(value):
            return "N/A"
        
        sign = "+" if value > 0 else ""
        return f"{sign}{value:.{decimals}f}%"
    
    @staticmethod
    def format_large_number(value: float, decimals: int = 1) -> str:
        """Format large number with appropriate suffix."""
        if pd.isna(value):
            return "N/A"
        
        abs_value = abs(value)
        
        if abs_value >= 1_000_000_000:
            return f"{value/1_000_000_000:.{decimals}f}B"
        elif abs_value >= 1_000_000:
            return f"{value/1_000_000:.{decimals}f}M"
        elif abs_value >= 1_000:
            return f"{value/1_000:.{decimals}f}K"
        else:
            return f"{value:.0f}"
    
    @staticmethod
    def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M") -> str:
        """Format datetime object as string."""
        if pd.isna(dt):
            return "N/A"
        
        return dt.strftime(format_str)
    
    @staticmethod
    def create_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive summary statistics for DataFrame."""
        summary = {
            'shape': df.shape,
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isna().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary['numeric_summary'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                '25%': float(df[col].quantile(0.25)),
                '50%': float(df[col].quantile(0.5)),
                '75%': float(df[col].quantile(0.75)),
                'max': float(df[col].max()),
                'skew': float(df[col].skew()),
                'kurtosis': float(df[col].kurtosis())
            }
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            summary['categorical_summary'][col] = {
                'unique_count': int(df[col].nunique()),
                'top_values': df[col].value_counts().head(5).to_dict()
            }
        
        return summary


# Singleton instances for easy access
market_helpers = MarketDataHelpers()
data_helpers = DataStorageHelpers()
stat_helpers = StatisticalHelpers()
ts_helpers = TimeSeriesHelpers()
validation_helpers = ValidationHelpers()
format_helpers = FormattingHelpers()
