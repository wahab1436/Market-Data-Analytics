#!/usr/bin/env python3
"""
Market Insight Platform - Main Entry Point
Local-First MVP | Batch-Driven Analytics
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
from src.models.explainability import ModelExplainability
from src.dashboard.app import create_app


class MarketInsightPlatform:
    """Main orchestrator for the Market Insight Platform."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the platform with configuration."""
        self.config = self.load_config(config_path)
        self.logger = setup_logger(self.config)
        self.mode = None
        
        # Initialize paths
        self.setup_directories()
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)
    
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
            
            # Verify we have enough data for modeling
            data_summary = {}
            for symbol, df in features_data.items():
                data_summary[symbol] = len(df)
                self.logger.info(f"{symbol}: {len(df)} records after feature engineering")
            
            if all(count < 30 for count in data_summary.values()):
                self.logger.warning("Insufficient data for model training. Need at least 30 records per symbol.")
                self.logger.warning("Consider:")
                self.logger.warning("  1. Changing config outputsize to 'full'")
                self.logger.warning("  2. Extending date_range start date")
                self.logger.warning("  3. Reducing moving_averages window sizes")
            
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
            self.logger.info(f"Symbols processed: {len(features_data)}")
            for symbol, count in data_summary.items():
                self.logger.info(f"  - {symbol}: {count} records")
            self.logger.info(f"Analysis modules: {len([r for r in analysis_results.values() if r])}")
            self.logger.info(f"Models trained: Regression={bool(reg_results)}, KNN={bool(knn_results)}, XGBoost={bool(xgb_results)}")
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