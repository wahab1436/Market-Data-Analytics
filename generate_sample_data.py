"""
Improved Sample Data Generator
Creates realistic market data with proper time series characteristics
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Create directories
Path("data/raw").mkdir(parents=True, exist_ok=True)

# Configuration
start_date = datetime(2025, 12, 1)
end_date = datetime(2026, 1, 29)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Realistic base prices and characteristics for each symbol
symbols_config = {
    "AAPL": {
        "base_price": 258.0,
        "drift": 0.0003,  # Slight upward trend
        "volatility": 0.02,  # 2% daily volatility
        "volume_mean": 50000000,
        "volume_std": 15000000
    },
    "MSFT": {
        "base_price": 425.0,
        "drift": 0.0002,
        "volatility": 0.018,
        "volume_mean": 25000000,
        "volume_std": 8000000
    },
    "GOOGL": {
        "base_price": 175.0,
        "drift": 0.0001,
        "volatility": 0.022,
        "volume_mean": 20000000,
        "volume_std": 6000000
    }
}

print("=" * 60)
print("IMPROVED SAMPLE DATA GENERATOR")
print("=" * 60)
print(f"Date Range: {start_date.date()} to {end_date.date()}")
print(f"Trading Days: {len(dates)}")
print(f"Symbols: {', '.join(symbols_config.keys())}")
print("=" * 60)

for symbol, config in symbols_config.items():
    print(f"\nGenerating data for {symbol}...")
    
    # Set seed for reproducibility but make it symbol-specific
    np.random.seed(hash(symbol) % 2**32)
    
    # Initialize price series with geometric brownian motion
    base_price = config['base_price']
    drift = config['drift']
    volatility = config['volatility']
    
    # Generate price path
    prices = [base_price]
    for i in range(1, len(dates)):
        # Geometric Brownian Motion
        random_shock = np.random.normal(0, 1)
        price_change = prices[-1] * (drift + volatility * random_shock)
        new_price = prices[-1] + price_change
        
        # Add occasional larger moves (market events)
        if np.random.random() < 0.05:  # 5% chance of significant event
            event_magnitude = np.random.choice([-1, 1]) * np.random.uniform(0.02, 0.04)
            new_price *= (1 + event_magnitude)
        
        prices.append(max(new_price, base_price * 0.7))  # Floor at 70% of base
    
    # Generate OHLCV data
    time_series = {}
    
    for idx, date in enumerate(dates):
        close_price = prices[idx]
        
        # Generate realistic OHLC
        # Open slightly different from previous close
        if idx == 0:
            open_price = close_price * np.random.uniform(0.995, 1.005)
        else:
            gap = np.random.normal(0, 0.005)  # Overnight gap
            open_price = prices[idx-1] * (1 + gap)
        
        # Intraday range based on volatility
        daily_range = close_price * np.random.uniform(0.005, 0.025)
        
        high_price = max(open_price, close_price) + daily_range * np.random.uniform(0.3, 0.7)
        low_price = min(open_price, close_price) - daily_range * np.random.uniform(0.3, 0.7)
        
        # Ensure high >= close >= low and high >= open >= low
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Generate realistic volume
        base_volume = config['volume_mean']
        volume_std = config['volume_std']
        
        # Volume tends to be higher on larger price moves
        price_change_pct = abs(close_price - open_price) / open_price
        volume_multiplier = 1 + price_change_pct * 5  # Volume increases with volatility
        
        volume = int(np.random.normal(base_volume * volume_multiplier, volume_std))
        volume = max(volume, base_volume // 2)  # Floor at half of base
        
        # Weekend adjustment - lower volume on Fridays
        if date.weekday() == 4:  # Friday
            volume = int(volume * 0.85)
        
        # Create time series entry
        date_str = date.strftime('%Y-%m-%d')
        time_series[date_str] = {
            "1. open": f"{open_price:.4f}",
            "2. high": f"{high_price:.4f}",
            "3. low": f"{low_price:.4f}",
            "4. close": f"{close_price:.4f}",
            "5. volume": str(volume)
        }
    
    # Create full API response format
    data = {
        "Meta Data": {
            "1. Information": "Daily Prices (open, high, low, close) and Volumes",
            "2. Symbol": symbol,
            "3. Last Refreshed": end_date.strftime('%Y-%m-%d'),
            "4. Output Size": "Compact",
            "5. Time Zone": "US/Eastern"
        },
        "Time Series (Daily)": time_series
    }
    
    # Calculate statistics for reporting
    price_start = prices[0]
    price_end = prices[-1]
    price_return = ((price_end - price_start) / price_start) * 100
    price_volatility = np.std([prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]) * np.sqrt(252) * 100
    
    # Save to file
    output_file = Path(f"data/raw/{symbol}_sample.json")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  ✓ Created: {output_file}")
    print(f"    - Data points: {len(time_series)}")
    print(f"    - Price range: ${min(prices):.2f} - ${max(prices):.2f}")
    print(f"    - Total return: {price_return:+.2f}%")
    print(f"    - Annualized volatility: {price_volatility:.1f}%")
    print(f"    - Avg volume: {np.mean([int(v['5. volume']) for v in time_series.values()]):,.0f}")

print("\n" + "=" * 60)
print("SAMPLE DATA GENERATION COMPLETE!")
print("=" * 60)
print("\nNext steps:")
print("1. Run: python main.py --mode batch")
print("2. Run: python main.py --mode dashboard")
print("=" * 60)