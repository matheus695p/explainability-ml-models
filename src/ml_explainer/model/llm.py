import os

import openai
import typing as tp
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from langchain.chains.llm import LLMChain
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate

from ml_explainer.model.shap_values import generate_msg_by_index, generate_shap_message

_ = load_dotenv(find_dotenv())
openai.api_base = os.environ["OPENAI_API_KEY"]
openai.api_key = os.environ["OPENAI_API_BASE"]


def answer_question_with_llm(
    template: PromptTemplate,
    feature_description_msg: str,
    feature_importance_msg: str,
    shap_prediction_msg: str,
    llm: LLMChain,
    question: str,
):
    """
    Generate a final answer to a question using an LLM (Language Model).

    Args:
        template (str): The template for generating prompts.
        feature_description_msg (str): Message explaining feature names.
        feature_importance_msg (str): Message explaining feature importance.
        shap_prediction_msg (str): Message explaining predictions using SHAP values.
        llm: The language model (LLM) to use for generating responses.
        question (str): The question to be answered.

    Returns:
        str: The final answer to the question.

    Example:
        final_answer = answer_question_with_llm(
            template,
            feature_description_msg,
            feature_importance_msg,
            shap_prediction_msg,
            llm,
            question
        )
        print(final_answer)
    """
    # Format the template with provided messages
    template_formatted = template.format(
        feature_description_msg=feature_description_msg,
        feature_importance_msg=feature_importance_msg,
        shap_prediction_msg=shap_prediction_msg,
    )

    # Initialize a PromptTemplate
    prompt = PromptTemplate(input_variables=["question"], template=template_formatted)

    # Initialize an LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Generate the response based on the provided question
    result = chain.run(question=question)

    # Combine the result into the final answer
    final_answer = result + "\n"

    return final_answer


def generate_explainability_report(
    shap_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    parameters: tp.Dict[str, tp.Any],
    report_params: tp.Dict[str, tp.Dict],
):
    """
    Generate an explainability report and final answer for a machine learning model.

    Args:
        shap_df (pd.DataFrame): DataFrame containing SHAP values.
        feature_importance_df (pd.DataFrame): DataFrame containing feature importance values.
        parameters (tp.Dict[str, tp.Any]): Parameters for the report generation.
        report_params (tp.Dict[str, tp.Any]): Parameters specific to the report.

    Returns:
        tp.Dict[str, str]: A dictionary containing the report and final answer.

    Example:
        shap_df = pd.DataFrame(...)  # Provide SHAP values DataFrame
        feature_importance_df = pd.DataFrame(...)  # Provide feature importance DataFrame
        parameters = {...}  # Provide parameters
        report_params = {...}  # Provide report parameters

        result = generate_explainability_report(shap_df, feature_importance_df, parameters, report_params)
        print(result["report"])
        print(result["final_answer"])
    """
    number_of_observations_to_explain = report_params["number_of_observations_to_explain"]
    feature_description = parameters["feature_description"]

    # define llm
    llm = OpenAI(temperature=0)

    # template information
    template = parameters["conversation_chain"]["prompt_template"]
    questions = report_params["questions"]
    starter_questions = report_params["starter_questions"]

    # starting messages
    feature_importance_df["description"] = feature_importance_df.index.map(feature_description)
    feature_description_msg = generate_msg_by_index(
        df=feature_importance_df, column="description", additional_msg=""
    )
    feature_importance_msg = generate_msg_by_index(
        df=feature_importance_df, column="feature_importance", additional_msg="[%]"
    )

    # Initial part of the report
    final_answer = ""
    for question_key, question in starter_questions.items():
        answer = answer_question_with_llm(
            template=template,
            feature_description_msg=feature_description_msg,
            feature_importance_msg=feature_importance_msg,
            shap_prediction_msg="",
            llm=llm,
            question=question,
        )
        final_answer += answer

    # answer all the particular question to the inference
    for prediction_index in range(number_of_observations_to_explain):
        shap_info = shap_df.iloc[prediction_index].to_frame().T
        shap_prediction_msg = generate_shap_message(shap_info=shap_info)

        for question_key, question in questions.items():
            final_answer += f"Prediction inference number {prediction_index} {question_key}" + "\n"
            answer = answer_question_with_llm(
                template=template,
                feature_description_msg=feature_description_msg,
                feature_importance_msg=feature_importance_msg,
                shap_prediction_msg=shap_prediction_msg,
                llm=llm,
                question=question,
            )
            final_answer += answer

    formatting_question = report_params["formatting_question"] + "\n" + final_answer

    report = answer_question_with_llm(
        template=template,
        feature_description_msg="",
        feature_importance_msg="",
        shap_prediction_msg="",
        llm=llm,
        question=formatting_question,
    )
    return dict(
        report=report,
        final_answer=final_answer,
    )
