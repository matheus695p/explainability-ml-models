from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_shap_information


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=generate_shap_information,
                inputs=["regressor", "X_train", "X_test"],
                outputs=dict(
                    shap_values="shap_values",
                    fig_shap="shap_values_figure",
                ),
                name="shap_information",
            ),
        ]
    )
