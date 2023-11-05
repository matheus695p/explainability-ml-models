import typing as tp

import pandas as pd
import shap
from sklearn.base import RegressorMixin

from ml_explainer.model.shap_values import (
    calculate_feature_importance_df,
    create_shap_dataframe,
    generate_shap_beeswarm_plot,
)


def generate_shap_information(
    regressor: RegressorMixin, X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tp.Dict[str, tp.Any]:
    """
    Generate SHAP values and visualization for a regression model.

    This function calculates SHAP values for a given regression model's predictions on a test dataset
    and generates a beeswarm plot to visualize the SHAP values.

    Parameters:
    - regressor (RegressorMixin): The regression model to explain using SHAP values.
    - X_train (pd.DataFrame): The training data used to fit the regression model.
    - X_test (pd.DataFrame): The test data for which SHAP values are calculated.

    Returns:
    - dict: A dictionary containing the following elements:
        - 'shap_values' (pd.DataFrame): A DataFrame containing base values and SHAP values for each sample.
        - 'fig_shap' (matplotlib.figure.Figure): The generated SHAP beeswarm plot.

    Example usage:
    ```python
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    # Create and fit a regression model
    regressor = RandomForestRegressor(n_estimators=100)
    regressor.fit(X_train, y_train)

    # Generate SHAP values and visualization
    shap_info = generate_shap_information(regressor, X_train, X_test)

    # Access SHAP values DataFrame and display the plot
    shap_values_df = shap_info['shap_values']
    fig_shap = shap_info['fig_shap']
    plt.show()  # Display the SHAP beeswarm plot
    ```
    """
    # resampling to speed up computing
    X_train = X_train.sample(7000, replace=True)

    # shap explainer
    features = list(X_train.columns)
    explainer = shap.Explainer(regressor.predict, X_train.astype(float))

    # shap values
    shap_values_train = explainer(X_train.astype(float))
    shap_values_test = explainer(X_test.astype(float))

    # shap dataframes
    shap_values_df_train = create_shap_dataframe(shap_values=shap_values_train, features=features)
    shap_values_df_test = create_shap_dataframe(shap_values=shap_values_test, features=features)

    # feature importance df
    feature_importance_df = calculate_feature_importance_df(
        shap_values=shap_values_train, features=features
    )

    # shap figures
    fig_test = generate_shap_beeswarm_plot(shap_values=shap_values_train, max_display=20)
    fig_train = generate_shap_beeswarm_plot(shap_values=shap_values_test, max_display=20)

    return dict(
        shap_values_df_train=shap_values_df_train,
        shap_values_df_test=shap_values_df_test,
        fig_shap_test=fig_test,
        fig_shap_train=fig_train,
        feature_importance=feature_importance_df,
    )
