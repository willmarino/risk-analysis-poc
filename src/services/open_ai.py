import pandas as pd
from sklearn.inspection import permutation_importance
from openai import OpenAI
# from tabulate import to_markdown

client = OpenAI()

def generate_explanation(rf, train_inputs, train_outputs, random_input, random_output):
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

    # https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    importances = pd.DataFrame({
        "feature": feature_names,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)

    # https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance
    perm_importance = permutation_importance(rf, train_inputs, train_outputs, n_repeats=10)
    perm_importances = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm_importance.importances_mean,
        "importance_std": perm_importance.importances_std
    }).sort_values("importance_mean", ascending=False)

    completion = client.chat.completions.create(
        model="gpt-4o",
        store=True,
        messages=[
            {
                "role": "user",
                "content": f"""
                    Hello, I need some help generating an explanation of why a
                    series of numbers combined into a vector embedding is indicative of a certain outcome.
                    The vector embedding is this {random_input},
                    where each number in the list corresponds by index to the features in {feature_names}.
                    So, the first number in {random_input} is a representation of Annual_Revenue,
                    and the second number is a representation of Debt_To_Income_Ratio, and so on.
                    The importance of each of these variables in predicting the outcome {random_output}
                    is given by these two analyses of feature importance.
                    Non-permutated feature importances:
                    {importances.to_markdown(index=False)}
                    Permutated feature importances:
                    {perm_importances.to_markdown(index=False)}

                    Given all of this data, can you explain to me why certain
                    variables in the vector embedding could have caused the outcome {random_output}?
                """
            }
        ]
    )
    
    return completion.choices[0].message.content


