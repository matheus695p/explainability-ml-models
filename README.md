# Machine learning model explainer 

## Overview

Welcome to the `ml_explainer` project, a tool for generating explainability reports for machine learning models. This Kedro project was inspired by the [spaceflights tutorial](https://docs.kedro.org/en/stable/tutorial/spaceflights_tutorial.html) and was generated using `Kedro 0.18.14`.

### Project Objective

The primary goal of this project is to create an ML package that can automatically generate comprehensive explanability reports for a wide range of machine learning models. These reports follow a specific template and utilize GenAI to provide detailed insights into the decision-making processes of the models.

### Key Features

- **Explainability Reports**: Generate detailed and informative reports that explain the inner workings of machine learning models.

- **Template-Based**: Follow a predefined template to ensure consistency and clarity in the generated reports.

- **GenAI Integration**: Leverage GenAI to enhance the explanations by providing insights and context from a vast knowledge base.

## Example of Explanability report generated using GenAI

[Interpretability Report Output Example](https://github.com/matheus695p/explainability-ml-models/blob/main/ExplainabilityReport.md)

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository to your local machine.
2. Create a `.env` file in the project's root directory.
3. Add your OpenAI API keys to the `.env` file as follows:

```env
OPENAI_API_BASE="YOUR_OPENAI_API_BASE"
OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"
```

## How to Install Dependencies

Declare project dependencies in the following files:

- `src/requirements.txt`: For `pip` installation.
- `src/environment.yml`: For `conda` installation.

To install dependencies, follow the instructions below:

### Using pip

Navigate to the project directory and run the following command:

```bash
pip install -r src/requirements.txt
```

### Generate a GenAI interpretability report
```
kedro run
```

Specifically to the report:


```
kedro run --pipeline explainer --nodes=explainability_report
```
