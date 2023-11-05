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


def create_shap_dataframe(shap_values: tp.List[list], X_train: pd.DataFrame):
    """
    Create a DataFrame from a list of SHAP values.

    Parameters:
      shap_values (list of numpy.ndarray): The list of SHAP values.
      X_train (pandas.DataFrame): The original training data with column names.

    Returns:
    - pandas.DataFrame: The resulting DataFrame containing base values and SHAP values.
    """
    base_values = [
        pd.DataFrame([shap_values[i].base_values], columns=["base_value"])
        for i in range(len(shap_values))
    ]
    df_base_value = pd.concat(base_values, axis=0).reset_index(drop=True)

    shaps_df = [
        pd.DataFrame([shap_values[i].values], columns=X_train.columns)
        for i in range(len(shap_values))
    ]
    shap_df = pd.concat(shaps_df, axis=0, ignore_index=False).reset_index(drop=True)

    df = pd.concat([df_base_value, shap_df], axis=1, ignore_index=False).reset_index(drop=True)

    return df