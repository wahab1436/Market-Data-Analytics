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
    
    def run_batch_pipeline(self):
        """Execute the complete batch pipeline."""
        self.logger.info("Starting batch pipeline execution")
        
        try:
            # 1. Data Ingestion
            self.logger.info("Step 1: Data Ingestion")
            fetcher = DataFetcher(self.config, self.logger)
            raw_data = fetcher.fetch_all_symbols()
            
            # 2. Data Cleaning
            self.logger.info("Step 2: Data Cleaning")
            cleaner = DataCleaner(self.config, self.logger)
            clean_data = cleaner.process_all_symbols(raw_data)
            
            # 3. Feature Engineering
            self.logger.info("Step 3: Feature Engineering")
            engineer = FeatureEngineer(self.config, self.logger)
            features_data = engineer.create_features(clean_data)
            
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
                analysis_results[name] = analyzer.analyze(features_data)
            
            # 5. Model Training
            self.logger.info("Step 5: Model Training")
            
            # Regression Models
            reg_models = RegressionModels(self.config, self.logger)
            reg_results = reg_models.train_and_evaluate(features_data)
            
            # KNN Similarity
            knn_model = KNNSimilarity(self.config, self.logger)
            knn_results = knn_model.train_and_evaluate(features_data)
            
            # XGBoost Model
            xgb_model = XGBoostModel(self.config, self.logger)
            xgb_results = xgb_model.train_and_evaluate(features_data)
            
            # 6. Explainability
            self.logger.info("Step 6: Model Explainability")
            
            # Only run explainability if XGBoost trained successfully
            explainability_results = {}
            if xgb_results and 'model' in xgb_results and xgb_results['model'] is not None:
                explainer = ModelExplainability(self.config, self.logger)
                explainability_results = explainer.compute_shap(
                    xgb_results['model'], 
                    xgb_results['X_test'], 
                    xgb_results['feature_names']
                )
            else:
                self.logger.warning("Skipping explainability - no trained XGBoost models available")
            
            self.logger.info("Batch pipeline completed successfully")
            
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
        
        # Verify precomputed data exists
        gold_path = Path(self.config['paths']['gold_data'])
        if not any(gold_path.glob("*.parquet")):
            self.logger.error("No precomputed data found. Run batch pipeline first.")
            sys.exit(1)
        
        # Create and run dashboard
        app = create_app(self.config)
        app.run(
            debug=False,
            host='127.0.0.1',
            port=8050,
            dev_tools_ui=False,
            dev_tools_props_check=False
        )
    
    def run(self, mode: str):
        """Run the platform in specified mode."""
        self.mode = mode
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
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Initialize and run platform
    platform = MarketInsightPlatform(args.config)
    platform.run(args.mode)


if __name__ == "__main__":
    main()