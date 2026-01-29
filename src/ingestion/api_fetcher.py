"""
Data Ingestion Layer - API Fetcher with Caching
Alpha Vantage API implementation with strict rate limiting
"""

import os
import time
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class DataFetcher:
    """Fetches market data from Alpha Vantage API with caching and rate limiting."""
    
    def __init__(self, config: Dict[str, Any], logger):
        """Initialize fetcher with configuration."""
        self.config = config
        self.logger = logger
        self.api_key = self._get_api_key()
        self.base_url = config['data']['api']['base_url']
        self.rate_limit_seconds = config['data']['api']['rate_limit_seconds']
        self.max_retries = config['data']['api']['max_retries']
        self.cache_duration = timedelta(hours=config['data']['api']['cache_duration_hours'])
        self.raw_data_path = Path(config['paths']['raw_data'])
        self.last_request_time = 0
        
    def _get_api_key(self) -> str:
        """Get API key from environment variable."""
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            self.logger.error("ALPHA_VANTAGE_API_KEY environment variable not set")
            raise ValueError("API key not found in environment variables")
        return api_key
    
    def _respect_rate_limit(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_seconds:
            sleep_time = self.rate_limit_seconds - time_since_last
            self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_cache_key(self, symbol: str) -> str:
        """Generate cache key for symbol."""
        cache_key = f"{symbol}_{self.config['data']['date_range']['start']}_{self.config['data']['date_range']['end']}"
        return hashlib.md5(cache_key.encode()).hexdigest()[:16]
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cached data is still valid."""
        if not cache_file.exists():
            return False
        
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age < self.cache_duration
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.exceptions.ConnectionError, 
                                      requests.exceptions.Timeout))
    )
    def _fetch_from_api(self, symbol: str) -> Dict[str, Any]:
        """Fetch data from Alpha Vantage API."""
        self._respect_rate_limit()
        
        params = {
            'function': self.config['data']['api']['function'],
            'symbol': symbol,
            'outputsize': self.config['data']['api']['outputsize'],
            'datatype': self.config['data']['api']['datatype'],
            'apikey': self.api_key
        }
        
        self.logger.info(f"Fetching data for {symbol} from Alpha Vantage")
        
        try:
            response = requests.get(
                self.base_url,
                params=params,
                timeout=30,
                headers={'User-Agent': 'MarketInsightPlatform/1.0'}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for API errors
                if 'Error Message' in data:
                    self.logger.error(f"API error for {symbol}: {data['Error Message']}")
                    raise ValueError(f"API error: {data['Error Message']}")
                
                if 'Note' in data:
                    self.logger.warning(f"API note for {symbol}: {data['Note']}")
                    # This is usually a rate limit warning
                    time.sleep(60)  # Wait longer if rate limited
                
                return data
                
            elif response.status_code == 429:
                self.logger.warning(f"Rate limited for {symbol}, waiting 60 seconds")
                time.sleep(60)
                raise requests.exceptions.RetryError("Rate limited")
                
            else:
                response.raise_for_status()
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for {symbol}: {e}")
            raise
    
    def _save_to_cache(self, symbol: str, data: Dict[str, Any]):
        """Save raw data to cache."""
        cache_key = self._get_cache_key(symbol)
        cache_file = self.raw_data_path / f"{symbol}_{cache_key}.json"
        
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.debug(f"Cached data for {symbol} to {cache_file}")
    
    def _load_from_cache(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Load data from cache if valid."""
        cache_key = self._get_cache_key(symbol)
        cache_file = self.raw_data_path / f"{symbol}_{cache_key}.json"
        
        if self._is_cache_valid(cache_file):
            with open(cache_file, 'r') as f:
                data = json.load(f)
            self.logger.info(f"Loaded cached data for {symbol}")
            return data
        
        return None
    
    def fetch_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch data for a single symbol with caching."""
        # Try cache first
        cached_data = self._load_from_cache(symbol)
        if cached_data:
            return cached_data
        
        # Fetch from API
        try:
            data = self._fetch_from_api(symbol)
            self._save_to_cache(symbol, data)
            return data
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None
    
    def fetch_all_symbols(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """Fetch data for all configured symbols."""
        symbols = self.config['data']['symbols']
        results = {}
        
        self.logger.info(f"Fetching data for {len(symbols)} symbols")
        
        for symbol in symbols:
            data = self.fetch_symbol(symbol)
            results[symbol] = data
            
            # Be extra careful with rate limits
            if symbol != symbols[-1]:  # Don't wait after last symbol
                time.sleep(1)
        
        # Check for failures
        successful = [s for s, d in results.items() if d is not None]
        failed = [s for s, d in results.items() if d is None]
        
        if failed:
            self.logger.warning(f"Failed to fetch data for symbols: {failed}")
        
        if not successful:
            raise ValueError("No data was successfully fetched")
        
        self.logger.info(f"Successfully fetched data for {len(successful)} symbols")
        return results
    
    def validate_api_response(self, data: Dict[str, Any]) -> bool:
        """Validate API response structure."""
        required_keys = ['Meta Data', 'Time Series (Daily)']
        
        if not all(key in data for key in required_keys):
            self.logger.error(f"Missing required keys in API response")
            return False
        
        time_series = data.get('Time Series (Daily)', {})
        if not time_series:
            self.logger.error("Empty time series data")
            return False
        
        # Check for required OHLCV fields in first entry
        first_date = next(iter(time_series))
        first_entry = time_series[first_date]
        
        required_fields = ['1. open', '2. high', '3. low', '4. close', '5. volume']
        if not all(field in first_entry for field in required_fields):
            self.logger.error("Missing OHLCV fields in time series")
            return False
        
        return True