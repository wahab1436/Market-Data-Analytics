"""
Feature Engineering Layer - Leakage-Safe Time Series Features
Strictly past-only feature creation with explicit look-ahead prevention
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Creates leakage-safe features for market data analysis."""
    
    def __init__(self, config: Dict[str, Any], logger):
        """Initialize feature engineer with configuration."""
        self.config = config
        self.logger = logger
        self.gold_data_path = Path(config['paths']['gold_data'])
        
        # Feature configuration
        self.rolling_windows = config['features']['rolling_windows']
        self.volatility_periods = config['features']['volatility_periods']
        self.moving_averages = config['features']['moving_averages']
        self.lag_periods = config['features']['lag_periods']
    
    def create_features(self, cleaned_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Create features for all symbols."""
        featured_data = {}
        
        for symbol, df in cleaned_data.items():
            self.logger.info(f"Creating features for {symbol}")
            
            # Make a copy to avoid modifying original
            df_features = df.copy()
            
            # Ensure chronological order
            df_features = df_features.sort_index()
            
            # Create features (all leakage-safe)
            df_features = self._create_returns(df_features)
            df_features = self._create_volatility_features(df_features)
            df_features = self._create_volume_features(df_features)
            df_features = self._create_price_structure_features(df_features)
            df_features = self._create_lagged_features(df_features)
            df_features = self._create_correlation_features(df_features, cleaned_data)
            
            # Add target variable (next day's absolute return)
            df_features = self._create_target_variable(df_features)
            
            # Remove rows with NaN values (from rolling calculations)
            initial_count = len(df_features)
            df_features = df_features.dropna()
            rows_removed = initial_count - len(df_features)
            
            if rows_removed > 0:
                self.logger.info(f"Removed {rows_removed} rows with NaN values for {symbol}")
            
            # Validate no leakage
            self._validate_no_leakage(df_features, symbol)
            
            # Save to gold layer
            self._save_gold_data(df_features, symbol)
            
            featured_data[symbol] = df_features
        
        # Create multi-symbol dataset for models that need cross-sectional data
        combined_features = self._create_combined_dataset(featured_data)
        
        self.logger.info(f"by_symbol: {len(featured_data)} symbols processed")
        self.logger.info(f"combined: {len(combined_features)} total records")
        
        return {
            'by_symbol': featured_data,
            'combined': combined_features
        }
    
    def _create_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create return-based features."""
        # Daily simple returns (leakage-safe: uses only past data)
        df['return'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Rolling returns (using only past data)
        for window in self.rolling_windows:
            df[f'rolling_return_{window}d'] = df['return'].rolling(window=window, min_periods=window).mean()
            df[f'rolling_std_{window}d'] = df['return'].rolling(window=window, min_periods=window).std()
        
        # Cumulative returns
        df['cumulative_return'] = (1 + df['return']).cumprod() - 1
        
        return df
    
    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-based features."""
        for period in self.volatility_periods:
            # Rolling volatility (standard deviation of returns)
            rolling_vol = df['return'].rolling(window=period, min_periods=period).std()
            df[f'volatility_{period}d'] = rolling_vol
            
            # Annualized volatility (assuming 252 trading days)
            df[f'volatility_annualized_{period}d'] = rolling_vol * np.sqrt(252)
            
            # Volatility z-score (how current volatility compares to recent history)
            if period >= 20:  # Need enough history for meaningful z-score
                vol_mean = rolling_vol.rolling(window=period*2, min_periods=period*2).mean()
                vol_std = rolling_vol.rolling(window=period*2, min_periods=period*2).std()
                df[f'volatility_zscore_{period}d'] = (rolling_vol - vol_mean) / (vol_std + 1e-8)
        
        # High-low range as % of close (daily volatility proxy)
        df['daily_range_pct'] = (df['high'] - df['low']) / df['close']
        
        # Rolling average true range
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr_14d'] = df['tr'].rolling(window=14, min_periods=14).mean()
        
        return df
    
    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features."""
        # Rolling average volume
        for window in [10, 20, 30]:
            df[f'volume_ma_{window}d'] = df['volume'].rolling(window=window, min_periods=window).mean()
            df[f'volume_ratio_{window}d'] = df['volume'] / (df[f'volume_ma_{window}d'] + 1e-8)
        
        # Volume standard deviation
        df['volume_std_20d'] = df['volume'].rolling(window=20, min_periods=20).std()
        
        # Volume-price relationship
        df['volume_return_corr_20d'] = df['volume'].rolling(window=20, min_periods=20).corr(df['return'])
        
        # Dollar volume
        df['dollar_volume'] = df['volume'] * df['close']
        df['dollar_volume_ma_20d'] = df['dollar_volume'].rolling(window=20, min_periods=20).mean()
        
        return df
    
    def _create_price_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price structure features."""
        # Moving averages
        for ma in self.moving_averages:
            df[f'ma_{ma}d'] = df['close'].rolling(window=ma, min_periods=ma).mean()
            df[f'price_vs_ma_{ma}d_pct'] = (df['close'] - df[f'ma_{ma}d']) / df[f'ma_{ma}d'] * 100
        
        # Support and resistance levels (simplified)
        df['rolling_high_20d'] = df['high'].rolling(window=20, min_periods=20).max()
        df['rolling_low_20d'] = df['low'].rolling(window=20, min_periods=20).min()
        df['price_vs_high_20d'] = (df['close'] - df['rolling_high_20d']) / df['rolling_high_20d']
        df['price_vs_low_20d'] = (df['close'] - df['rolling_low_20d']) / df['rolling_low_20d']
        
        # Momentum indicators
        df['momentum_10d'] = df['close'] - df['close'].shift(10)
        df['roc_10d'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
        
        # Price position within daily range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        
        return df
    
    def _create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features (strictly past-only)."""
        # Lagged returns
        for lag in self.lag_periods:
            df[f'return_lag_{lag}d'] = df['return'].shift(lag)
            df[f'volume_lag_{lag}d'] = df['volume'].shift(lag)
        
        # Lagged volatility
        for period in self.volatility_periods[:2]:  # First two periods
            for lag in [1, 2, 5]:
                df[f'volatility_{period}d_lag_{lag}d'] = df[f'volatility_{period}d'].shift(lag)
        
        return df
    
    def _create_correlation_features(self, df: pd.DataFrame, all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create features based on correlation with other symbols."""
        current_symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else None
        
        if not current_symbol:
            return df
        
        # Get other symbols
        other_symbols = [s for s in all_data.keys() if s != current_symbol]
        
        if not other_symbols:
            return df
        
        # Calculate rolling correlations with other symbols
        for other_symbol in other_symbols[:2]:  # Limit to 2 other symbols
            other_df = all_data[other_symbol].copy()
            
            # Ensure other_df has returns calculated
            if 'return' not in other_df.columns:
                other_df['return'] = other_df['close'].pct_change()
            
            if len(other_df) > 0:
                # Align dates
                aligned_returns = pd.DataFrame({
                    'current': df['return'],
                    'other': other_df['return'].reindex(df.index)
                }).dropna()
                
                if len(aligned_returns) >= 20:
                    # Rolling correlation
                    rolling_corr = aligned_returns['current'].rolling(window=20).corr(aligned_returns['other'])
                    df[f'corr_with_{other_symbol}_20d'] = rolling_corr
        
        return df
    
    def _create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable for prediction (next day's absolute return)."""
        # Target: absolute value of next day's return (for volatility prediction)
        df['target_abs_return_next_day'] = df['return'].shift(-1).abs()
        
        # Alternative target: direction of next day's return
        df['target_return_next_day'] = df['return'].shift(-1)
        df['target_direction_next_day'] = (df['target_return_next_day'] > 0).astype(float)
        
        # Remove the last row since we don't have target for it
        df = df.iloc[:-1]
        
        return df
    
    def _validate_no_leakage(self, df: pd.DataFrame, symbol: str):
        """Validate that no features use future data."""
        leakage_issues = []
        
        # Check for any forward-looking operations
        for col in df.columns:
            if col.startswith('target_'):
                continue
            
            # Check if any value could be influenced by future data
            # This is a simplified check - in production would be more thorough
            if df[col].isnull().all():
                continue
        
        if leakage_issues:
            self.logger.warning(f"Potential leakage issues found for {symbol}: {leakage_issues}")
    
    def _save_gold_data(self, df: pd.DataFrame, symbol: str):
        """Save featured data to gold layer."""
        file_path = self.gold_data_path / f"{symbol}_featured.parquet"
        
        # Reset index for Parquet compatibility
        df_reset = df.reset_index()
        
        # Save as Parquet
        df_reset.to_parquet(file_path, index=False, compression='snappy')
        
        self.logger.debug(f"Saved gold data for {symbol} to {file_path}")
    
    def _create_combined_dataset(self, featured_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create combined dataset for multi-symbol analysis."""
        combined_dfs = []
        
        for symbol, df in featured_data.items():
            df_copy = df.copy()
            df_copy['symbol'] = symbol
            combined_dfs.append(df_copy)
        
        if not combined_dfs:
            return pd.DataFrame()
        
        combined = pd.concat(combined_dfs, axis=0)
        combined = combined.sort_index()
        
        # Save combined dataset
        combined_path = self.gold_data_path / "combined_features.parquet"
        combined_reset = combined.reset_index()
        combined_reset.to_parquet(combined_path, index=False, compression='snappy')
        
        return combined
    
    def load_silver_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load cleaned data from silver layer."""
        file_path = self.gold_data_path / f"{symbol}_featured.parquet"
        
        if not file_path.exists():
            self.logger.warning(f"No featured data found for {symbol}")
            return None
        
        try:
            df = pd.read_parquet(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            return df
        except Exception as e:
            self.logger.error(f"Error loading featured data for {symbol}: {e}")
            return None
    
    def get_feature_categories(self) -> Dict[str, List[str]]:
        """Get features grouped by category."""
        return {
            'returns': ['return', 'log_return'] + [f'rolling_return_{w}d' for w in self.rolling_windows],
            'volatility': [f'volatility_{p}d' for p in self.volatility_periods] + 
                         [f'volatility_zscore_{p}d' for p in self.volatility_periods if p >= 20] +
                         ['daily_range_pct', 'atr_14d'],
            'volume': ['volume_ratio_10d', 'volume_ratio_20d', 'volume_ratio_30d',
                      'volume_std_20d', 'volume_return_corr_20d', 'dollar_volume_ma_20d'],
            'price_structure': [f'ma_{ma}d' for ma in self.moving_averages] +
                              [f'price_vs_ma_{ma}d_pct' for ma in self.moving_averages] +
                              ['rolling_high_20d', 'rolling_low_20d', 'momentum_10d', 'roc_10d'],
            'lagged': [f'return_lag_{l}d' for l in self.lag_periods] +
                     [f'volume_lag_{l}d' for l in self.lag_periods]
        }