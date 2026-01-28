import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

Path("data/raw").mkdir(parents=True, exist_ok=True)

start_date = datetime(2025, 12, 1)
end_date = datetime(2026, 1, 29)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

symbols = {"AAPL": 258, "MSFT": 425, "GOOGL": 175}

for symbol, base_price in symbols.items():
    np.random.seed(hash(symbol) % 2**32)
    time_series = {}
    
    for date in dates:
        price = base_price * (1 + np.random.normal(0, 0.02))
        time_series[date.strftime('%Y-%m-%d')] = {
            "1. open": f"{price * 0.99:.4f}",
            "2. high": f"{price * 1.01:.4f}",
            "3. low": f"{price * 0.98:.4f}",
            "4. close": f"{price:.4f}",
            "5. volume": str(int(np.random.uniform(40000000, 80000000)))
        }
    
    data = {
        "Meta Data": {
            "1. Information": "Daily Prices",
            "2. Symbol": symbol,
            "3. Last Refreshed": end_date.strftime('%Y-%m-%d'),
            "4. Output Size": "Compact",
            "5. Time Zone": "US/Eastern"
        },
        "Time Series (Daily)": time_series
    }
    
    with open(f"data/raw/{symbol}_sample.json", 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Created sample data for {symbol}")

print("\nSample data generation complete!")