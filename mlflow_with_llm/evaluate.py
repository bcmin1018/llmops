import mlflow
import pandas as pd

logged_model = "runs:/8b00411f051a45fa8d3b4760b64985e8/model"

article_text = """
An MLflow Project is a format for packaging data science code in a reusable and reproducible way.
The MLflow Projects component includes an API and command-line tools for running projects, which
also integrate with the Tracking component to automatically record the parameters and git commit
of your source code for reproducibility.

This article describes the format of an MLflow Project and how to run an MLflow project remotely
using the MLflow CLI, which makes it easy to vertically scale your data science code.
"""
question = "What is an MLflow project?"

data = pd.DataFrame(
    {
        "article": [article_text],
        "question": [question],
        "ground_truth": [
            article_text
        ],  # used for certain evaluation metrics, such as ROUGE score
    }
)

with mlflow.start_run():
    results = mlflow.evaluate(
        model=logged_model,
        data=data,
        targets="ground_truth",
        model_type="text-summarization",
    )

eval_table = results.tables["eval_results_table"]
print(f"See evaluation table below: \n{eval_table}")
