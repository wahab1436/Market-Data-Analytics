"""
Logging Utility - Professional Logging Configuration
Sanitized logs with no secrets or sensitive data
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any
import structlog


def setup_logger(config: Dict[str, Any]) -> structlog.BoundLogger:
    """Configure structured logging for the application."""
    
    log_config = config.get('logging', {})
    level = log_config.get('level', 'INFO')
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_config.get('file', 'market_insight.log')
    
    # Create log directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()  # Structured logging
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format=log_format,
        level=getattr(logging, level),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Get logger
    logger = structlog.get_logger()
    
    # Log startup information (sanitized)
    logger.info(
        "Logger initialized",
        level=level,
        log_file=str(log_path.absolute()),
        symbols=config['data']['symbols'],
        date_range=config['data']['date_range']
    )
    
    return logger


class SanitizedLogger:
    """Logger wrapper that ensures no sensitive data is logged."""
    
    def __init__(self, base_logger):
        self.logger = base_logger
        self.sensitive_patterns = [
            'apikey', 'api_key', 'password', 'secret', 'token',
            'ALPHA_VANTAGE', 'FINNHUB', 'MARKETSTACK',
            'key=', 'token=', 'secret=', 'password='
        ]
    
    def sanitize(self, message: str) -> str:
        """Remove sensitive information from log messages."""
        if not isinstance(message, str):
            return str(message)
        
        sanitized = message
        for pattern in self.sensitive_patterns:
            # Case-insensitive replacement
            import re
            sanitized = re.sub(
                pattern, 
                '[REDACTED]', 
                sanitized, 
                flags=re.IGNORECASE
            )
        
        # Also check for any patterns that look like API keys (long strings)
        # Remove any strings longer than 20 chars that might be keys
        words = sanitized.split()
        cleaned_words = []
        for word in words:
            if len(word) > 30 and '=' in word:
                # Looks like a key-value pair with long value
                parts = word.split('=', 1)
                if len(parts) == 2 and len(parts[1]) > 20:
                    cleaned_words.append(f"{parts[0]}=[REDACTED]")
                else:
                    cleaned_words.append(word)
            else:
                cleaned_words.append(word)
        
        return ' '.join(cleaned_words)
    
    def info(self, message: str, **kwargs):
        """Log info message with sanitization."""
        safe_message = self.sanitize(message)
        safe_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, str):
                safe_kwargs[key] = self.sanitize(value)
            else:
                safe_kwargs[key] = value
        
        self.logger.info(safe_message, **safe_kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with sanitization."""
        safe_message = self.sanitize(message)
        safe_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, str):
                safe_kwargs[key] = self.sanitize(value)
            else:
                safe_kwargs[key] = value
        
        self.logger.warning(safe_message, **safe_kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with sanitization."""
        safe_message = self.sanitize(message)
        safe_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, str):
                safe_kwargs[key] = self.sanitize(value)
            else:
                safe_kwargs[key] = value
        
        self.logger.error(safe_message, **safe_kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with sanitization."""
        safe_message = self.sanitize(message)
        safe_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, str):
                safe_kwargs[key] = self.sanitize(value)
            else:
                safe_kwargs[key] = value
        
        self.logger.debug(safe_message, **safe_kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with sanitization."""
        safe_message = self.sanitize(message)
        safe_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, str):
                safe_kwargs[key] = self.sanitize(value)
            else:
                safe_kwargs[key] = value
        
        self.logger.critical(safe_message, **safe_kwargs)


def get_logger(name: str = None, config: Dict[str, Any] = None) -> structlog.BoundLogger:
    """Get a logger instance."""
    if config:
        return setup_logger(config)
    else:
        return structlog.get_logger(name)


# Example usage and testing
if __name__ == "__main__":
    # Test the logger
    test_config = {
        'logging': {
            'level': 'DEBUG',
            'file': 'test_log.log'
        },
        'data': {
            'symbols': ['AAPL', 'MSFT'],
            'date_range': {
                'start': '2023-01-01',
                'end': '2023-12-31'
            }
        }
    }
    
    logger = setup_logger(test_config)
    sanitized_logger = SanitizedLogger(logger)
    
    # Test logging (safe)
    logger.info("Application started")
    logger.info("Fetching data for symbols", symbols=['AAPL', 'MSFT'])
    
    # Test sanitization
    sanitized_logger.info("API key found: ALPHA_VANTAGE_API_KEY=abc123def456ghi789")
    sanitized_logger.info("Request to https://www.alphavantage.co/query?apikey=SECRET_KEY&symbol=AAPL")
    sanitized_logger.info("Finnhub token: xyz789")
    
    print("Logger test completed. Check test_log.log for output.")
