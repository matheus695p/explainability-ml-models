import logging
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml_explainer.model.object_inyection import load_estimator, load_object


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    y = data[parameters["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: Dict) -> RegressorMixin:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor1 = load_estimator(parameters["model1"])
    regressor2 = load_estimator(parameters["model2"])
    final_estimator = load_estimator(parameters["final_estimator"])
    scaler = load_object(parameters["scaler"])
    imputer = load_object(parameters["imputer"])

    regressor = build_model(
        imputer=imputer,
        scaler=scaler,
        model1=regressor1,
        model2=regressor2,
        final_estimator=final_estimator,
    )
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_model(regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series):
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)


def build_model(
    scaler: Any,
    imputer: Any,
    model1: RegressorMixin,
    model2: RegressorMixin,
    final_estimator: RegressorMixin = Ridge(),
):
    """
    Build a stacked ensemble regression model using two base regression models.

    This function creates a stacked ensemble regression model that combines the predictions
    of two base regression models (model1 and model2) after preprocessing the numerical data.

    Parameters:
    - model1 (RegressorMixin): The first base regression model to include in the ensemble.
    - model2 (RegressorMixin): The second base regression model to include in the ensemble.

    Returns:
    - StackingRegressor: A stacked ensemble regression model combining the two base models.

    Example usage:
    ```python
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    # Create base regression models
    model1 = RandomForestRegressor(n_estimators=100)
    model2 = LinearRegression()

    # Build a stacked ensemble regression model
    stacked_model = build_model(model1, model2)

    # Fit the ensemble model on training data and make predictions
    stacked_model.fit(X_train, y_train)
    predictions = stacked_model.predict(X_test)
    ```
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", imputer),
            ("scaler", scaler),
        ]
    )

    # Create the original regression pipeline
    regression_pipeline1 = Pipeline(
        [
            ("preprocessor", numeric_transformer),
            ("regressor1", model1),
        ]
    )

    # Create the original regression pipeline
    regression_pipeline2 = Pipeline(
        [
            ("preprocessor", numeric_transformer),
            ("regressor2", model2),
        ]
    )
    # Create a stacker pipeline that combines the original regression model and the new regression model
    stacker = StackingRegressor(
        estimators=[
            ("regressor1", regression_pipeline1),
            ("regressor2", regression_pipeline2),
        ],
        final_estimator=final_estimator,
    )

    return stacker
