#!/usr/bin/env python3
"""
Market Insight Platform - Main Entry Point
Local-First MVP | Batch-Driven Analytics
Optimized for compact API output (100 days)
FIXED: Proper handling of nested feature data structure
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger
from src.ingestion.api_fetcher import DataFetcher
from src.preprocessing.cleaner import DataCleaner
from src.features.market_features import FeatureEngineer
from src.analysis.price_analysis import PriceAnalysis
from src.analysis.volatility_analysis import VolatilityAnalysis
from src.analysis.volume_analysis import VolumeAnalysis
from src.analysis.similarity_analysis import SimilarityAnalysis
from src.models.regression import RegressionModels
from src.models.knn_similarity import KNNSimilarity
from src.models.xgboost_model import XGBoostModel
from src.models.explainability import SimilarityAnalysis as ModelExplainability
from src.dashboard.app import create_app


class MarketInsightPlatform:
    """Main orchestrator for the Market Insight Platform."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the platform with configuration."""
        self.config = self.load_config(config_path)
        
        # Optimize config for compact mode if needed
        self.optimize_for_compact_mode()
        
        self.logger = setup_logger(self.config)
        self.mode = None
        
        # Initialize paths
        self.setup_directories()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file with proper encoding."""
        try:
            # Try UTF-8 first (standard)
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except UnicodeDecodeError:
            # Fallback to UTF-8 with BOM
            try:
                with open(config_path, 'r', encoding='utf-8-sig') as f:
                    config = yaml.safe_load(f)
                return config
            except Exception:
                # Last resort: try latin-1
                try:
                    with open(config_path, 'r', encoding='latin-1') as f:
                        config = yaml.safe_load(f)
                    return config
                except Exception as e:
                    print(f"Error loading config: {e}")
                    print("Please ensure config file is saved in UTF-8 encoding")
                    sys.exit(1)
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)
    
    def optimize_for_compact_mode(self):
        """Optimize feature windows for compact API mode (100 days)."""
        if self.config.get('data', {}).get('api', {}).get('outputsize') == 'compact':
            print("INFO: Detected compact mode - optimizing feature windows")
            
            # Reduce moving average windows to fit in 100 days
            if 'features' in self.config:
                # Original windows might be [20, 50, 200]
                # Reduce to [10, 20, 50] for compact mode
                original_mas = self.config['features'].get('moving_averages', [20, 50, 200])
                if max(original_mas) > 50:
                    self.config['features']['moving_averages'] = [10, 20, 50]
                    print(f"INFO: Reduced moving averages from {original_mas} to [10, 20, 50]")
                
                # Reduce rolling windows
                original_windows = self.config['features'].get('rolling_windows', [10, 20, 30, 50])
                if max(original_windows) > 30:
                    self.config['features']['rolling_windows'] = [5, 10, 20, 30]
                    print(f"INFO: Reduced rolling windows from {original_windows} to [5, 10, 20, 30]")
    
    def setup_directories(self):
        """Create necessary directories."""
        paths = self.config['paths']
        for path_key in ['raw_data', 'silver_data', 'gold_data', 'models', 'artifacts', 'logs']:
            Path(paths[path_key]).mkdir(parents=True, exist_ok=True)
    
    def check_data_availability(self) -> Dict[str, bool]:
        """Check what data is available for dashboard."""
        gold_path = Path(self.config['paths']['gold_data'])
        artifacts_path = Path(self.config['paths']['artifacts'])
        
        return {
            'parquet_files': len(list(gold_path.glob("*.parquet"))) > 0,
            'analysis_files': len(list(artifacts_path.glob("*_analysis_results.pkl"))) > 0,
            'model_files': len(list(artifacts_path.glob("*_model_results.pkl"))) > 0,
            'parquet_count': len(list(gold_path.glob("*.parquet"))),
            'analysis_count': len(list(artifacts_path.glob("*_analysis_results.pkl"))),
            'model_count': len(list(artifacts_path.glob("*_model_results.pkl")))
        }
    
    def run_batch_pipeline(self):
        """Execute the complete batch pipeline."""
        self.logger.info("Starting batch pipeline execution")
        
        # Log configuration
        api_mode = self.config.get('data', {}).get('api', {}).get('outputsize', 'unknown')
        self.logger.info(f"API Output Size: {api_mode}")
        self.logger.info(f"Moving Averages: {self.config.get('features', {}).get('moving_averages', [])}")
        
        try:
            # 1. Data Ingestion
            self.logger.info("Step 1: Data Ingestion")
            fetcher = DataFetcher(self.config, self.logger)
            raw_data = fetcher.fetch_all_symbols()
            
            if not raw_data:
                self.logger.error("No data fetched. Check API connection and credentials.")
                sys.exit(1)
            
            # 2. Data Cleaning
            self.logger.info("Step 2: Data Cleaning")
            cleaner = DataCleaner(self.config, self.logger)
            clean_data = cleaner.process_all_symbols(raw_data)
            
            if not clean_data:
                self.logger.error("Data cleaning failed. Check raw data quality.")
                sys.exit(1)
            
            # 3. Feature Engineering
            self.logger.info("Step 3: Feature Engineering")
            engineer = FeatureEngineer(self.config, self.logger)
            features_data = engineer.create_features(clean_data)
            
            if not features_data:
                self.logger.error("Feature engineering failed. Check feature configuration.")
                sys.exit(1)
            
            # =====================================================================
            # FIX: Access by_symbol from nested structure
            # =====================================================================
            # Verify we have enough data for modeling
            data_summary = {}
            for symbol, df in features_data['by_symbol'].items():
                data_summary[symbol] = len(df)
                self.logger.info(f"{symbol}: {len(df)} records after feature engineering")
            
            # Provide helpful warnings based on data availability
            min_records = min(data_summary.values()) if data_summary else 0
            if min_records < 10:
                self.logger.error("CRITICAL: Less than 10 records per symbol. Pipeline cannot continue.")
                self.logger.error("Solutions:")
                self.logger.error("  1. Change outputsize to 'full' in config.yaml")
                self.logger.error("  2. Extend date_range.start to earlier date")
                self.logger.error("  3. Current mode is too restrictive for analysis")
                sys.exit(1)
            elif min_records < 30:
                self.logger.warning("WARNING: Limited data available. Some models may not train.")
                self.logger.warning(f"Records available: {min_records} (recommended: 30+)")
                self.logger.warning("Consider changing outputsize to 'full' for better results.")
            
            # 4. Analysis
            self.logger.info("Step 4: Market Analysis")
            analyses = {
                'price': PriceAnalysis(self.config, self.logger),
                'volatility': VolatilityAnalysis(self.config, self.logger),
                'volume': VolumeAnalysis(self.config, self.logger),
                'similarity': SimilarityAnalysis(self.config, self.logger)
            }
            
            analysis_results = {}
            for name, analyzer in analyses.items():
                self.logger.info(f"Running {name} analysis")
                try:
                    analysis_results[name] = analyzer.analyze(features_data)
                except Exception as e:
                    self.logger.warning(f"{name} analysis failed: {e}")
                    analysis_results[name] = {}
            
            # 5. Model Training
            self.logger.info("Step 5: Model Training")
            
            # Regression Models
            reg_results = {}
            try:
                reg_models = RegressionModels(self.config, self.logger)
                reg_results = reg_models.train_and_evaluate(features_data)
            except Exception as e:
                self.logger.warning(f"Regression training failed: {e}")
            
            # KNN Similarity
            knn_results = {}
            try:
                knn_model = KNNSimilarity(self.config, self.logger)
                knn_results = knn_model.train_and_evaluate(features_data)
            except Exception as e:
                self.logger.warning(f"KNN training failed: {e}")
            
            # XGBoost Model
            xgb_results = {}
            try:
                xgb_model = XGBoostModel(self.config, self.logger)
                xgb_results = xgb_model.train_and_evaluate(features_data)
            except Exception as e:
                self.logger.warning(f"XGBoost training failed: {e}")
            
            # 6. Explainability
            self.logger.info("Step 6: Model Explainability")
            
            # Only run explainability if XGBoost trained successfully
            explainability_results = {}
            if xgb_results and 'model' in xgb_results and xgb_results.get('model') is not None:
                try:
                    explainer = ModelExplainability(self.config, self.logger)
                    explainability_results = explainer.compute_shap(
                        xgb_results['model'],
                        xgb_results['X_test'],
                        xgb_results['feature_names']
                    )
                except Exception as e:
                    self.logger.warning(f"Explainability computation failed: {e}")
            else:
                self.logger.warning("Skipping explainability - no trained XGBoost models available")
            
            self.logger.info("Batch pipeline completed successfully")
            
            # Print summary
            self.logger.info("=" * 60)
            self.logger.info("BATCH PIPELINE SUMMARY")
            self.logger.info("=" * 60)
            self.logger.info(f"API Mode: {api_mode}")
            self.logger.info(f"Symbols processed: {len(data_summary)}")
            for symbol, count in data_summary.items():
                self.logger.info(f"  - {symbol}: {count} records")
            self.logger.info(f"Analysis modules completed: {len([r for r in analysis_results.values() if r])}/4")
            self.logger.info(f"Models trained: Regression={bool(reg_results)}, KNN={bool(knn_results)}, XGBoost={bool(xgb_results)}")
            self.logger.info("=" * 60)
            
            # Provide recommendations based on results
            if min_records < 50:
                self.logger.info("RECOMMENDATION: For better model performance:")
                self.logger.info("  - Change config.yaml: outputsize: 'full'")
                self.logger.info("  - This will provide 500+ records instead of ~50")
                self.logger.info("=" * 60)
            
            return {
                'features': features_data,
                'analysis': analysis_results,
                'models': {
                    'regression': reg_results,
                    'knn': knn_results,
                    'xgboost': xgb_results
                },
                'explainability': explainability_results
            }
        
        except Exception as e:
            self.logger.error(f"Batch pipeline failed: {e}")
            raise
    
    def run_dashboard(self):
        """Launch the Plotly Dash dashboard."""
        self.logger.info("Starting dashboard server")
        
        # Check data availability
        data_status = self.check_data_availability()
        
        self.logger.info("=" * 60)
        self.logger.info("DATA AVAILABILITY CHECK")
        self.logger.info("=" * 60)
        self.logger.info(f"Feature files (.parquet): {data_status['parquet_count']} files")
        self.logger.info(f"Analysis files (.pkl): {data_status['analysis_count']} files")
        self.logger.info(f"Model files (.pkl): {data_status['model_count']} files")
        self.logger.info("=" * 60)
        
        # Improved check - allow dashboard to start even with partial data
        if not data_status['parquet_files']:
            self.logger.warning("WARNING: No feature files found in data/gold/")
            self.logger.warning("WARNING: Dashboard will show limited or no data")
            self.logger.warning("WARNING: Run: python main.py --mode batch")
            self.logger.warning("")
            
            # Ask user if they want to continue
            response = input("Continue starting dashboard anyway? (y/N): ")
            if response.lower() != 'y':
                self.logger.info("Dashboard startup cancelled")
                sys.exit(0)
        
        if not data_status['analysis_files']:
            self.logger.warning("WARNING: No analysis files found - some visualizations may be empty")
        
        if not data_status['model_files']:
            self.logger.warning("WARNING: No model files found - predictive tabs may be empty")
        
        # Create and run dashboard
        try:
            app = create_app(self.config)
            
            self.logger.info("=" * 60)
            self.logger.info("DASHBOARD SERVER STARTING")
            self.logger.info("=" * 60)
            self.logger.info("URL: http://127.0.0.1:8050/")
            self.logger.info("Press CTRL+C to stop the server")
            self.logger.info("=" * 60)
            
            app.run(
                debug=False,
                host='127.0.0.1',
                port=8050,
                dev_tools_ui=False,
                dev_tools_props_check=False
            )
        except Exception as e:
            self.logger.error(f"Dashboard failed to start: {e}")
            raise
    
    def run(self, mode: str):
        """Run the platform in specified mode."""
        self.mode = mode
        
        print("=" * 60)
        print(f"Market Insight Platform v{self.config['project']['version']}")
        print("=" * 60)
        print(f"Mode: {mode.upper()}")
        print(f"Symbols: {', '.join(self.config['data']['symbols'])}")
        print(f"Date Range: {self.config['data']['date_range']['start']} to {self.config['data']['date_range']['end']}")
        print(f"API Output: {self.config.get('data', {}).get('api', {}).get('outputsize', 'unknown')}")
        print("=" * 60)
        print()
        
        self.logger.info(f"Running Market Insight Platform in {mode} mode")
        
        if mode == 'batch':
            self.run_batch_pipeline()
        elif mode == 'dashboard':
            self.run_dashboard()
        else:
            self.logger.error(f"Invalid mode: {mode}. Use 'batch' or 'dashboard'")
            sys.exit(1)


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description='Market Insight Platform - Local Analytics MVP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run batch pipeline to fetch and process data
  python main.py --mode batch

  # Start dashboard to visualize data
  python main.py --mode dashboard

  # Use custom config file
  python main.py --mode batch --config custom_config.yaml

Notes:
  - Compact mode (outputsize: compact) provides ~100 days of data
  - Full mode (outputsize: full) provides up to 20 years of data
  - Compact mode works but with reduced feature windows
  - For best results, use full mode in config.yaml
"""
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['batch', 'dashboard'],
        required=True,
        help='Execution mode: "batch" for pipeline, "dashboard" for visualization'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    args = parser.parse_args()
    
    # Initialize and run platform
    try:
        platform = MarketInsightPlatform(args.config)
        platform.run(args.mode)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()