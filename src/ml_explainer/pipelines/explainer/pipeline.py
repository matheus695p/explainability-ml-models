from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_shap_information
from ml_explainer.model.llm import generate_explainability_report


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=generate_shap_information,
                inputs=["regressor", "X_train", "X_test"],
                outputs=dict(
                    shap_values_df_train="shap_values_df_train",
                    shap_values_df_test="shap_values_df_test",
                    fig_shap_test="fig_shap_test",
                    fig_shap_train="fig_shap_train",
                    feature_importance="feature_importance",
                ),
                name="shap_information",
            ),
            node(
                func=generate_explainability_report,
                inputs=dict(
                    shap_df="shap_values_df_test",
                    feature_importance_df="feature_importance",
                    parameters="params:chain_config",
                    report_params="params:report_information",
                ),
                outputs=dict(
                    report="shap_values_df_train",
                    final_answer="shap_values_df_test",
                ),
                name="explainability_report",
            ),
        ]
    )
