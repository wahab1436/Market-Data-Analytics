"""
Data Ingestion Layer - Unified Fetcher (API + Sample Data)
Supports both Alpha Vantage API and local sample JSON files
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
    """Fetches market data from API or local sample files with intelligent fallback."""
    
    def __init__(self, config: Dict[str, Any], logger):
        """Initialize fetcher with configuration."""
        self.config = config
        self.logger = logger
        
        # Determine data source
        self.source = config['data'].get('source', 'api').lower()
        
        # Common settings
        self.raw_data_path = Path(config['paths']['raw_data'])
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        
        # API-specific settings (only if using API)
        if self.source == 'api':
            self.api_key = self._get_api_key()
            if self.api_key:
                self.base_url = config['data']['api']['base_url']
                self.rate_limit_seconds = config['data']['api']['rate_limit_seconds']
                self.max_retries = config['data']['api']['max_retries']
                self.cache_duration = timedelta(hours=config['data']['api']['cache_duration_hours'])
                self.last_request_time = 0
                self.logger.info("✓ Initialized with API data source")
            else:
                self.logger.warning("⚠ API key not found, falling back to sample data")
                self.source = 'sample'
        
        if self.source == 'sample':
            self.logger.info("✓ Initialized with SAMPLE data source")
        
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment variable (optional for sample mode)."""
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            self.logger.warning("ALPHA_VANTAGE_API_KEY not set")
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
    
    # ==================== SAMPLE DATA METHODS ====================
    
    def _load_sample_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Load data from local sample JSON file."""
        sample_file = self.raw_data_path / f"{symbol}_sample.json"
        
        if not sample_file.exists():
            self.logger.error(f"✗ Sample file not found: {sample_file}")
            return None
        
        try:
            with open(sample_file, 'r') as f:
                data = json.load(f)
            self.logger.info(f"✓ Loaded sample data for {symbol}")
            return data
        except Exception as e:
            self.logger.error(f"✗ Failed to load sample data for {symbol}: {e}")
            return None
    
    # ==================== API METHODS ====================
    
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
        
        self.logger.info(f"Fetching data for {symbol} from Alpha Vantage API...")
        
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
                    self.logger.warning(f"API rate limit warning for {symbol}: {data['Note']}")
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
            self.logger.info(f"✓ Loaded cached API data for {symbol}")
            return data
        
        return None
    
    # ==================== UNIFIED INTERFACE ====================
    
    def fetch_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch data for a single symbol (from sample or API)."""
        
        # SAMPLE MODE
        if self.source == 'sample':
            return self._load_sample_data(symbol)
        
        # API MODE
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
            self.logger.error(f"✗ Failed to fetch API data for {symbol}: {e}")
            # Fallback to sample data if API fails
            self.logger.info(f"Attempting to load sample data as fallback...")
            return self._load_sample_data(symbol)
    
    def fetch_all_symbols(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """Fetch data for all configured symbols."""
        symbols = self.config['data']['symbols']
        results = {}
        
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Fetching data for {len(symbols)} symbols using {self.source.upper()} source")
        self.logger.info(f"{'='*60}")
        
        for symbol in symbols:
            data = self.fetch_symbol(symbol)
            results[symbol] = data
            
            # Rate limit only for API mode
            if self.source == 'api' and symbol != symbols[-1]:
                time.sleep(1)
        
        # Check for failures
        successful = [s for s, d in results.items() if d is not None]
        failed = [s for s, d in results.items() if d is None]
        
        if failed:
            self.logger.warning(f"✗ Failed to fetch data for symbols: {failed}")
        
        if not successful:
            raise ValueError("No data was successfully fetched")
        
        self.logger.info(f"✓ Successfully fetched data for {len(successful)}/{len(symbols)} symbols")
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
    
    def get_source_type(self) -> str:
        """Return current data source type."""
        return self.source