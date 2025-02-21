from openai import OpenAI

from .df_util import read_df_from_csv

client = OpenAI()

def generate_explanation(vector_input, output):
    print(f"Generating data explanation via OpenAI...")
    feature_importance_df = read_df_from_csv("feature_importance.csv")
    perm_feature_importance_df = read_df_from_csv("perm_feature_importance.csv")
    
    feature_names = read_df_from_csv("train_ve.csv").drop(columns=["Loan_ID", "Approval_Status", "zilliz_insertion_id"]).columns.tolist()

    completion = client.chat.completions.create(
        model="gpt-4o",
        store=True,
        messages=[
            {
                "role": "user",
                "content": f"""
                    Hello, I need some help generating an explanation of why a
                    series of numbers, representing small business loan data,
                    combined into a vector embedding is indicative of a certain outcome.
                    The vector embedding is this {vector_input},
                    where each number in the list corresponds by index to the features in {feature_names}.
                    So, the first number in {vector_input} is a representation of Annual_Revenue,
                    and the second number is a representation of Debt_To_Income_Ratio, and so on.
                    The importance of each of these variables in predicting the outcome {output}
                    is given by these two analyses of feature importance.
                    Non-permutated feature importances:
                    {feature_importance_df.to_markdown(index=False)}
                    Permutated feature importances:
                    {perm_feature_importance_df.to_markdown(index=False)}

                    Given all of this data, can you explain to me why certain
                    variables in the vector embedding could have caused the outcome {output}?

                    Keep in mind I want to gear this explanation towards use by an internal team of reviewers,
                    who want your help performing an initial valuation of a small business's records.

                """
            }
        ]
    )
    
    return completion.choices[0].message.content


