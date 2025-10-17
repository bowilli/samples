# models/model_wrappers.py
import pandas as pd
import numpy as np

class ModelWrapper:
    """A generic model wrapper to standardize the interface."""
    
    def __init__(self, model):
        self.model = model
    
    def train(self, features, target):
        """Train the model."""
        if hasattr(self.model, 'fit'):
            self.model.fit(features, target)
        else:
            raise NotImplementedError("Custom training logic is required for this model.")
    
    def predict(self, features):
        """Predict using the model."""
        if hasattr(self.model, 'predict'):
            return self.model.predict(features)
        else:
            raise NotImplementedError("Custom prediction logic is required for this model.")
    
    def get_model(self):
        return self.model
    
    def set_params(self, **params):
        """Set parameters on the underlying model."""
        if hasattr(self.model, 'set_params'):
            # Handle nested parameters for TransformedTargetRegressor
            if hasattr(self.model, 'regressor'):
                # For TransformedTargetRegressor, prefix parameters with 'regressor__'
                nested_params = {f'regressor__{key}': value for key, value in params.items()}
                return self.model.set_params(**nested_params)
            else:
                return self.model.set_params(**params)
        else:
            # For models without set_params, try setting attributes directly
            for key, value in params.items():
                setattr(self.model, key, value)
            return self
    
    def get_params(self, deep=True):
        """Get parameters from the underlying model."""
        if hasattr(self.model, 'get_params'):
            params = self.model.get_params(deep=deep)
            # Handle nested parameters for TransformedTargetRegressor
            if hasattr(self.model, 'regressor'):
                # Remove 'regressor__' prefix for consistency
                regressor_params = {key.replace('regressor__', ''): value 
                                  for key, value in params.items() 
                                  if key.startswith('regressor__')}
                return regressor_params
            else:
                return params
        else:
            # For models without get_params, return empty dict
            return {}

# Example of a custom model
class CustomModel:
    """A custom model without 'fit' and 'predict'."""
    
    def train_custom(self, features, target):
        # Custom training logic
        pass
    
    def predict_custom(self, features):
        # Custom prediction logic
        return [0] * len(features)  # Dummy prediction

class CustomModelWrapper(ModelWrapper):
    """Wrapper for the custom model."""
    
    def train(self, features, target):
        self.model.train_custom(features, target)
    
    def predict(self, features):
        return self.model.predict_custom(features)


class CatBoostWrapper(ModelWrapper):
    """Special wrapper for CatBoost that handles categorical features natively."""
    
    def __init__(self, model):
        super().__init__(model)
        self.feature_names = None
        self.cat_feature_indices = None
    
    def train(self, features, target, feature_names=None):
        """
        Train the CatBoost model with categorical features.
        Automatically detects categorical features based on data types.
        
        Args:
            features: NumPy array or DataFrame of features
            target: Target values
            feature_names: List of feature column names (optional)
        """
        # If features is a DataFrame, use it directly
        if isinstance(features, pd.DataFrame):
            # Make a copy to avoid modifying the original
            features = features.copy()
            self.feature_names = list(features.columns)
            
            # Automatically identify categorical features based on dtype
            # Check for object (string), bool, or category dtypes
            self.cat_feature_indices = [
                i for i, col in enumerate(features.columns) 
                if features[col].dtype in ['object', 'bool', 'category'] or 
                   col in ['returning_season', 'unscripted']  # These might be stored as int but are categorical
            ]
            
            if self.cat_feature_indices:
                # Fill NaN values in categorical columns with 'missing' string
                cat_cols = [features.columns[i] for i in self.cat_feature_indices]
                for col in cat_cols:
                    features[col] = features[col].fillna('missing')
                
                print(f"CatBoost detected categorical features at indices: {self.cat_feature_indices}")
                print(f"Categorical columns: {cat_cols}")
                # CatBoost can handle categorical features directly
                self.model.fit(features, target, cat_features=self.cat_feature_indices, verbose=False)
            else:
                print("No categorical features detected for CatBoost")
                self.model.fit(features, target, verbose=False)
        
        # If features is a numpy array and we have feature names
        elif feature_names is not None:
            self.feature_names = feature_names
            # Convert to DataFrame for easier handling
            features_df = pd.DataFrame(features, columns=feature_names)
            
            # Try to infer categorical features from the data
            self.cat_feature_indices = []
            for i, col in enumerate(feature_names):
                # Check if column has few unique values (likely categorical)
                # or if it's a known categorical column name
                unique_ratio = len(np.unique(features[:, i])) / len(features[:, i])
                if unique_ratio < 0.05 or col in ['gravity_buying_team', 'returning_season', 
                                                   'unscripted', 'os_production_type']:
                    self.cat_feature_indices.append(i)
            
            if self.cat_feature_indices:
                print(f"CatBoost detected categorical features at indices: {self.cat_feature_indices}")
                self.model.fit(features_df, target, cat_features=self.cat_feature_indices, verbose=False)
            else:
                print("No categorical features detected for CatBoost")
                self.model.fit(features_df, target, verbose=False)
        else:
            # No feature names provided, train without categorical features
            print("Warning: No feature names provided to CatBoost, treating all as numerical")
            self.model.fit(features, target, verbose=False)
    
    def predict(self, features):
        """Predict using the CatBoost model."""
        # If features is a numpy array and we have feature names, convert to DataFrame
        if isinstance(features, np.ndarray) and self.feature_names is not None:
            features = pd.DataFrame(features, columns=self.feature_names)
        
        # If features is a DataFrame and we have categorical indices, handle NaN values
        if isinstance(features, pd.DataFrame) and self.cat_feature_indices:
            features = features.copy()
            cat_cols = [features.columns[i] for i in self.cat_feature_indices]
            for col in cat_cols:
                features[col] = features[col].fillna('missing')
        
        return self.model.predict(features)
