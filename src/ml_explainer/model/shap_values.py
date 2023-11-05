import typing as tp

import matplotlib.pyplot as plt
import pandas as pd
import shap


def generate_shap_beeswarm_plot(shap_values: tp.List[list], max_display=20):
    """
    Generate a SHAP beeswarm plot with custom size adjustment.

    Parameters:
      shap_values (numpy.ndarray): The SHAP values to be visualized.
      max_display (int): Maximum number of data points to display.

    Returns:
    - matplotlib.figure.Figure: The generated figure.
    """
    fig, _ = plt.subplots()
    shap.plots.beeswarm(
        shap_values,
        max_display=max_display,
    )

    # Adjust the size of the plot
    original_size = fig.get_size_inches()
    fig.set_size_inches(2 * original_size[0], 2 * original_size[0] * 3 / 4)
    plt.tight_layout()
    # Return the figure
    return fig


def create_shap_dataframe(shap_values: tp.List[list], features: tp.List[str]):
    """
    Create a DataFrame from a list of SHAP values.

    Parameters:
      shap_values (list of numpy.ndarray): The list of SHAP values.
      X_train (pandas.DataFrame): The original training data with column names.

    Returns:
    - pandas.DataFrame: The resulting DataFrame containing base values and SHAP values.
    """
    shap_df = pd.DataFrame(shap_values.values, columns=features)
    shap_df = pd.concat(
        [pd.DataFrame(shap_values.base_values, columns=["base_value"]), shap_df], axis=1
    )
    return shap_df.reset_index(drop=True)


def calculate_feature_importance_df(shap_values: tp.List[list], features: tp.List[str]):
    """
    Calculate feature importances DataFrame from SHAP values.

    Parameters:
    - shap_values (shap.Explanation or pd.DataFrame): SHAP values or DataFrame with SHAP values.
    - X_train (pd.DataFrame): The original training data with column names.

    Returns:
    - pd.DataFrame: The feature importances DataFrame sorted by importance.
    """
    shap_df = create_shap_dataframe(shap_values=shap_values, features=features)

    shap_df[features] = shap_df[features].apply(abs)
    shap_df["shap_movement"] = shap_df[features].apply(abs).sum(axis=1)

    for col in features:
        shap_df[col] = shap_df[col] / shap_df["shap_movement"] * 100

    feature_importance_df = pd.DataFrame(shap_df[features].mean(), columns=["feature_importance"])
    feature_importance_df = feature_importance_df.sort_values(
        by=["feature_importance"], ascending=False
    )

    return feature_importance_df


def generate_shap_message(
    shap_info: pd.DataFrame,
) -> str:
    """
    Generate a message explaining the SHAP values in the shap_info DataFrame.

    Parameters:
    - shap_info (pd.DataFrame): DataFrame containing SHAP values and feature names.

    Returns:
    - str: The generated message explaining the SHAP values.
    """
    msg = ""
    shap_info = shap_info.reset_index(drop=True)
    for col in shap_info.columns:
        value = str(round(shap_info[col].iloc[0], 3))
        if value.startswith("-"):
            msg += f"{col} (has negative influence on the prediction) = {value}\n"
        else:
            msg += f"{col} (has positive influence on the prediction) = {value}\n"

    return msg


def generate_msg_by_index(df: pd.DataFrame, column: str, additional_msg: str = ""):
    """
    Generate a message explaining feature importance values in the provided DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing feature importance values.
    - additional_msg (str): An additional message to include after each feature's value.

    Returns:
    - str: The generated message explaining feature importance values.
    """
    msg = ""
    for feature in df.index:
        value = df.loc[feature, column]
        if isinstance(value, float) or isinstance(value, int):
            value = str(round(df.loc[feature, column], 3))
        if value.startswith("-"):
            msg += f"{feature} = {value} {additional_msg}\n"
        else:
            msg += f"{feature} = {value} {additional_msg}\n"

    return msg
