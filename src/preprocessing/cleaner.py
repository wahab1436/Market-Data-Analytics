"""
Data Cleaning & Validation Layer
Transforms raw API responses into clean, validated time series
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings


class DataCleaner:
    """Cleans and validates market data from raw API responses."""
    
    def __init__(self, config: Dict[str, Any], logger):
        """Initialize cleaner with configuration."""
        self.config = config
        self.logger = logger
        self.silver_data_path = Path(config['paths']['silver_data'])
        
    def process_symbol(self, symbol: str, raw_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Process raw data for a single symbol."""
        if not raw_data:
            self.logger.warning(f"No raw data for {symbol}")
            return None
        
        # Validate API response
        required_keys = ['Meta Data', 'Time Series (Daily)']
        if not all(key in raw_data for key in required_keys):
            self.logger.error(f"Invalid raw data structure for {symbol}")
            return None
        
        try:
            # Extract time series data
            time_series = raw_data['Time Series (Daily)']
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Rename columns to standard names
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high', 
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            }
            df = df.rename(columns=column_mapping)
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            df.index.name = 'date'
            
            # Sort by date
            df = df.sort_index()
            
            # Filter by date range
            start_date = pd.to_datetime(self.config['data']['date_range']['start'])
            end_date = pd.to_datetime(self.config['data']['date_range']['end'])
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            if df.empty:
                self.logger.warning(f"No data in date range for {symbol}")
                return None
            
            # Convert data types
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('int64')
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Handle missing values
            df = self._handle_missing_values(df, symbol)
            
            # Remove duplicates
            df = self._remove_duplicates(df, symbol)
            
            # Validate data quality
            is_valid = self._validate_data(df, symbol)
            
            if not is_valid:
                self.logger.error(f"Data validation failed for {symbol}")
                return None
            
            # Save to silver layer
            self._save_silver_data(df, symbol)
            
            self.logger.info(f"Processed {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}")
            return None
    
    def _handle_missing_values(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        initial_count = len(df)
        
        # Check for missing values
        missing_before = df.isnull().sum().sum()
        
        if missing_before > 0:
            self.logger.warning(f"Found {missing_before} missing values for {symbol}")
            
            # Forward fill for small gaps in price data
            price_cols = ['open', 'high', 'low', 'close']
            df[price_cols] = df[price_cols].ffill()
            
            # Backward fill if forward fill didn't work
            df[price_cols] = df[price_cols].bfill()
            
            # For volume, fill with 0
            df['volume'] = df['volume'].fillna(0)
            
            # Drop any remaining rows with missing values
            df = df.dropna()
            
            missing_after = df.isnull().sum().sum()
            rows_dropped = initial_count - len(df)
            
            if rows_dropped > 0:
                self.logger.warning(f"Dropped {rows_dropped} rows with missing values for {symbol}")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Remove duplicate entries."""
        initial_count = len(df)
        duplicates = df.index.duplicated(keep='first')
        
        if duplicates.any():
            df = df[~duplicates]
            self.logger.warning(f"Removed {duplicates.sum()} duplicate entries for {symbol}")
        
        return df
    
    def _validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Validate data quality and consistency."""
        # Check if DataFrame is empty
        if df.empty:
            self.logger.error(f"Empty DataFrame for {symbol}")
            return False
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'symbol']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.logger.error(f"Missing columns for {symbol}: {missing_columns}")
            return False
        
        # Check data types
        if not all(df[col].dtype in [np.float64, np.int64] for col in ['open', 'high', 'low', 'close']):
            self.logger.error(f"Invalid price data types for {symbol}")
            return False
        
        if df['volume'].dtype != np.int64:
            self.logger.error(f"Invalid volume data type for {symbol}")
            return False
        
        # Check for negative values
        price_cols = ['open', 'high', 'low', 'close']
        negative_prices = (df[price_cols] <= 0).any().any()
        
        if negative_prices:
            self.logger.error(f"Found non-positive prices for {symbol}")
            return False
        
        negative_volume = (df['volume'] < 0).any()
        if negative_volume:
            self.logger.error(f"Found negative volume for {symbol}")
            return False
        
        # Check for high-low consistency
        high_low_invalid = (df['high'] < df['low']).any()
        if high_low_invalid:
            self.logger.error(f"High < Low inconsistency for {symbol}")
            return False
        
        # Check for open-close range
        open_range_invalid = ((df['open'] < df['low']) | (df['open'] > df['high'])).any()
        close_range_invalid = ((df['close'] < df['low']) | (df['close'] > df['high'])).any()
        
        if open_range_invalid or close_range_invalid:
            self.logger.error(f"Open/Close outside High/Low range for {symbol}")
            return False
        
        # Check for large gaps (potential errors)
        returns = df['close'].pct_change().dropna()
        extreme_returns = (returns.abs() > 0.25).sum()  # >25% daily move
        
        if extreme_returns > 0:
            self.logger.warning(f"Found {extreme_returns} extreme returns (>25%) for {symbol}")
            # Don't fail for this, just warn
        
        return True
    
    def _save_silver_data(self, df: pd.DataFrame, symbol: str):
        """Save cleaned data to silver layer."""
        file_path = self.silver_data_path / f"{symbol}_cleaned.parquet"
        
        # Reset index for Parquet compatibility
        df_reset = df.reset_index()
        
        # Save as Parquet
        df_reset.to_parquet(file_path, index=False, compression='snappy')
        
        self.logger.debug(f"Saved silver data for {symbol} to {file_path}")
    
    def load_silver_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load cleaned data from silver layer."""
        file_path = self.silver_data_path / f"{symbol}_cleaned.parquet"
        
        if not file_path.exists():
            self.logger.warning(f"No silver data found for {symbol}")
            return None
        
        try:
            df = pd.read_parquet(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            return df
        except Exception as e:
            self.logger.error(f"Error loading silver data for {symbol}: {e}")
            return None
    
    def process_all_symbols(self, raw_data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Process all symbols from raw data."""
        processed_data = {}
        
        for symbol, data in raw_data.items():
            if data is None:
                continue
            
            df = self.process_symbol(symbol, data)
            if df is not None:
                processed_data[symbol] = df
        
        if not processed_data:
            raise ValueError("No symbols were successfully processed")
        
        self.logger.info(f"Successfully processed {len(processed_data)} symbols")
        return processed_data
    
    def combine_symbols(self, processed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine data from multiple symbols into single DataFrame."""
        combined_dfs = []
        
        for symbol, df in processed_data.items():
            df_copy = df.copy()
            # Ensure all required columns are present
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col not in df_copy.columns:
                    df_copy[col] = np.nan
            
            combined_dfs.append(df_copy)
        
        if not combined_dfs:
            return pd.DataFrame()
        
        combined = pd.concat(combined_dfs, axis=0)
        combined = combined.sort_index()
        
        return combined
