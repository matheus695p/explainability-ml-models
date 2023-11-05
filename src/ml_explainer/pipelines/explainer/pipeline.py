from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_shap_information


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=generate_shap_information,
                inputs=["regressor", "X_train", "X_test"],
                outputs=dict(
                    shap_values_df_train="shap_values_df_train",
                    shap_values_df_test="shap_values_df_test",
                    fig_shap_test="fig_test",
                    fig_shap_train="fig_train",
                    feature_importance="feature_importance",
                ),
                name="shap_information",
            ),
        ]
    )
