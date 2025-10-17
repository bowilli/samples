"""
Intelligent Director Model Predictor

Provides routing and prediction utilities for intelligent director models that
automatically select the appropriate model based on director fee history availability.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional


class IntelligentDirectorPredictor:
    """
    A predictor that routes titles to appropriate director models based on
    whether the directors have previous fee history.
    """

    def __init__(self):
        self.models = {}
        self.feature_configs = {}
        self.is_trained = False

    def register_models(self, model_cache: Dict[str, Any]):
        """
        Register trained models from model cache.

        Args:
            model_cache: Dictionary of cached models from main.py training
        """
        # Extract intelligent director models
        for model_key, model_data in model_cache.items():
            if '_intelligent' in model_key and model_data['target'] == 'director_efc_usd':
                subset_type = model_data['subset_type']  # 'no_fee_history' or 'with_fee_history'
                base_name = model_key.replace('_intelligent', '')

                if subset_type not in self.models:
                    self.models[subset_type] = {}
                    self.feature_configs[subset_type] = {}

                self.models[subset_type][base_name] = model_data['model']
                self.feature_configs[subset_type][base_name] = {
                    'features_encoded': model_data['features_encoded'],
                    'features_original': model_data['features_original'],
                    'set_name': model_data['set_name']
                }

        self.is_trained = len(self.models) > 0

    def predict_single_title(self, title_data: pd.Series, model_name: str = 'catboost') -> Dict[str, Any]:
        """
        Predict director fee for a single title using intelligent routing.

        Args:
            title_data: Series containing title features
            model_name: Name of the model to use (e.g., 'catboost', 'random_forest')

        Returns:
            Dictionary with prediction, confidence, and routing information
        """
        if not self.is_trained:
            raise ValueError("Models not registered. Call register_models() first.")

        # Determine which model to use based on fee history availability
        has_fee_history = pd.notna(title_data.get('mean_director_previous_director_efc_usd'))
        subset_type = 'with_fee_history' if has_fee_history else 'no_fee_history'

        # Find matching model
        matching_models = []
        for model_key in self.models.get(subset_type, {}):
            if model_name in model_key.lower():
                matching_models.append(model_key)

        if not matching_models:
            available_models = list(self.models.get(subset_type, {}).keys())
            raise ValueError(f"No {model_name} model found for {subset_type}. Available: {available_models}")

        # Use the first matching model (could be extended to select based on feature set)
        selected_model_key = matching_models[0]
        model = self.models[subset_type][selected_model_key]

        # Prepare features (simplified version - in practice would need full feature engineering pipeline)
        try:
            prediction = model.predict([title_data.values.reshape(1, -1)])[0] if hasattr(model, 'predict') else None

            return {
                'prediction': prediction,
                'model_used': selected_model_key,
                'subset_type': subset_type,
                'has_fee_history': has_fee_history,
                'confidence': None  # Could be extended with prediction intervals
            }
        except Exception as e:
            return {
                'prediction': None,
                'error': str(e),
                'model_used': selected_model_key,
                'subset_type': subset_type,
                'has_fee_history': has_fee_history
            }

    def predict_batch(self, titles_data: pd.DataFrame, model_name: str = 'catboost') -> pd.DataFrame:
        """
        Predict director fees for multiple titles using intelligent routing.

        Args:
            titles_data: DataFrame containing title features
            model_name: Name of the model to use

        Returns:
            DataFrame with predictions and routing information
        """
        if not self.is_trained:
            raise ValueError("Models not registered. Call register_models() first.")

        results = []

        for idx, title_data in titles_data.iterrows():
            result = self.predict_single_title(title_data, model_name)
            result['title_id'] = idx
            results.append(result)

        return pd.DataFrame(results)

    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of registered models."""
        summary = {
            'is_trained': self.is_trained,
            'model_counts': {},
            'available_subsets': list(self.models.keys())
        }

        for subset_type, models in self.models.items():
            summary['model_counts'][subset_type] = len(models)

        return summary


def create_intelligent_prediction_demo(model_cache: Dict[str, Any],
                                     sample_data: pd.DataFrame,
                                     n_samples: int = 5) -> None:
    """
    Create a demonstration of intelligent director prediction routing.

    Args:
        model_cache: Dictionary of cached models from main.py
        sample_data: Sample title data for demonstration
        n_samples: Number of samples to demonstrate
    """
    print("=" * 60)
    print("INTELLIGENT DIRECTOR PREDICTION DEMO")
    print("=" * 60)

    # Initialize predictor
    predictor = IntelligentDirectorPredictor()
    predictor.register_models(model_cache)

    # Show model summary
    summary = predictor.get_model_summary()
    print(f"\nModel Summary:")
    print(f"  Trained: {summary['is_trained']}")
    print(f"  Available subsets: {summary['available_subsets']}")
    for subset, count in summary['model_counts'].items():
        print(f"  Models for {subset}: {count}")

    if not summary['is_trained']:
        print("\n⚠️  No intelligent models found in cache")
        return

    # Sample titles for demonstration
    sample_titles = sample_data.sample(min(n_samples, len(sample_data)))

    print(f"\nDemonstrating prediction routing for {len(sample_titles)} sample titles:")
    print("-" * 60)

    for idx, (title_idx, title_data) in enumerate(sample_titles.iterrows()):
        has_fees = pd.notna(title_data.get('mean_director_previous_director_efc_usd'))
        actual_fee = title_data.get('director_efc_usd', 'N/A')

        print(f"\nTitle {idx + 1}: {title_data.get('titlename', 'Unknown')[:40]}...")
        print(f"  Actual director fee: ${actual_fee:,.0f}" if pd.notna(actual_fee) else f"  Actual director fee: {actual_fee}")
        print(f"  Has fee history: {has_fees}")

        # This would need the full feature engineering pipeline to work properly
        # For demo purposes, we just show the routing logic
        subset_type = 'with_fee_history' if has_fees else 'no_fee_history'
        print(f"  → Would route to: {subset_type} models")

        # Show available models for this routing
        available_models = list(predictor.models.get(subset_type, {}).keys())
        if available_models:
            print(f"  → Available models: {len(available_models)}")
            for model_key in available_models[:2]:  # Show first 2
                print(f"    - {model_key}")
            if len(available_models) > 2:
                print(f"    - ... and {len(available_models) - 2} more")

    print(f"\n{'=' * 60}")
    print("DEMO COMPLETE")
    print(f"{'=' * 60}")
