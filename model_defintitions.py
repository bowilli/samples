
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from .model_wrappers import ModelWrapper, CustomModel, CustomModelWrapper, CatBoostWrapper
from catboost import CatBoostRegressor
import numpy as np

def linear_regression():
    """Define a linear regression model with log-transformed target."""
    base_model = LinearRegression()
    transformed_model = TransformedTargetRegressor(
        regressor=base_model,
        func=np.log1p,  # log(1 + x) to handle zeros
        inverse_func=np.expm1  # exp(x) - 1 to inverse the transformation
    )
    return ModelWrapper(transformed_model)

def random_forest():
    """Define a random forest model with log-transformed target."""
    # Note: n_estimators, max_depth, min_samples_split, min_samples_leaf, criterion will be set via hyperparameter tuning
    base_model = RandomForestRegressor(
        random_state=42,  # For reproducibility
        n_jobs=-1  # Use all available cores
    )
    transformed_model = TransformedTargetRegressor(
        regressor=base_model,
        func=np.log1p,
        inverse_func=np.expm1
    )
    return ModelWrapper(transformed_model)

def custom_model():
    """Define a custom model."""
    return CustomModelWrapper(CustomModel())

def catboost():
    """Define a CatBoost model with log-transformed target and native categorical handling."""
    # Note: loss_function, iterations, learning_rate, depth, l2_leaf_reg will be set via hyperparameter tuning
    base_model = CatBoostRegressor(
        random_seed=42,
        verbose=False  # Reduce output noise during training
    )
    transformed_model = TransformedTargetRegressor(
        regressor=base_model,
        func=np.log1p,
        inverse_func=np.expm1
    )
    return CatBoostWrapper(transformed_model)
