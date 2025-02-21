# import pandas as pd
# from sklearn.inspection import permutation_importance
from openai import OpenAI

from ..models.random_forest import importances, perm_importances

client = OpenAI()

def generate_explanation(vector_input, output):
    feature_names = [
        "Annual_Revenue",
        "Debt_To_Income_Ratio", 
        "Credit_Score",
        "Loan_Amount_Requested",
        "Loan_Term_Months",
        "Interest_Rate",
        "Past_Loan_Defaults",
        "Business_Category_Construction",
        "Business_Category_Food & Beverage",
        "Business_Category_Healthcare",
        "Business_Category_Manufacturing",
        "Business_Category_Retail",
        "Business_Category_Technology"
    ]

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
                    {importances.to_markdown(index=False)}
                    Permutated feature importances:
                    {perm_importances.to_markdown(index=False)}

                    Given all of this data, can you explain to me why certain
                    variables in the vector embedding could have caused the outcome {output}?

                    Keep in mind I want to gear this explanation towards use by an internal team of reviewers,
                    who want your help performing an initial valuation of a small business's records.

                """
            }
        ]
    )
    
    return completion.choices[0].message.content


