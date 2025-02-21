import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from ..services.zilliz import fetch_vectors

# Load train and val dataframes from zilliz
train_ve = fetch_vectors("sbl_train", 0, 1000)
train_inputs = pd.DataFrame(list(map(lambda x: x["vector"], train_ve)))
train_outputs = list(map(lambda x: x["status"], train_ve))

rf = RandomForestClassifier()

rf.fit(train_inputs.values, train_outputs)

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