import os
import joblib
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV

from .zilliz import fetch_vectors
from .df_util import read_df_from_csv, write_df_to_csv


def get_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(script_dir, "..", "models", "random_forest.joblib")
    if Path(model_path).exists():
        print("Loading model from local storage...")
        rf = joblib.load(model_path)
    
    else: # train new model
        print("No model found in local storage, training new model...")

        # Get training data
        train_ve = fetch_vectors("sbl_train", 0, 1000)
        train_inputs = pd.DataFrame(list(map(lambda x: x["vector"], train_ve)))
        train_outputs = list(map(lambda x: x["status"], train_ve))

        # Create RF model using grid search, bumped accuracy up about 5%
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        grid_search = GridSearchCV(
            RandomForestClassifier(),
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )

        grid_search.fit(train_inputs, train_outputs)
        rf = grid_search.best_estimator_

        joblib.dump(rf, model_path)
    
    # This should probably be its own function
    # https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    # https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance
    # Run feature-importance-analysis to try to understand which variables are high-impact
    # Save as dataframes so they can be quickly loaded
    feature_names = read_df_from_csv("train_ve.csv").drop(columns=["Loan_ID", "Approval_Status", "zilliz_insertion_id"]).columns.tolist()
    
    importances_path = os.path.join(script_dir, "..", "..", "csv_data", "feature_importance.csv")
    if not Path(importances_path).exists():
        importances = pd.DataFrame({
            "feature": feature_names,
            "importance": rf.feature_importances_
        }).sort_values("importance", ascending=False)

        write_df_to_csv(importances, "feature_importance.csv")

    perm_importances_path = os.path.join(script_dir, "..", "..", "csv_data", "perm_feature_importance.csv")
    if not Path(perm_importances_path).exists():
        
        # This is dupped above
        train_ve = fetch_vectors("sbl_train", 0, 1000)
        train_inputs = pd.DataFrame(list(map(lambda x: x["vector"], train_ve)))
        train_outputs = list(map(lambda x: x["status"], train_ve))

        perm_importance = permutation_importance(rf, train_inputs, train_outputs, n_repeats=10)
        perm_importances = pd.DataFrame({
            "feature": feature_names,
            "importance_mean": perm_importance.importances_mean,
            "importance_std": perm_importance.importances_std
        }).sort_values("importance_mean", ascending=False)

        write_df_to_csv(perm_importances, "perm_feature_importance.csv")

    return rf