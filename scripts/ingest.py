import os
import pandas as pd
from scipy import stats
from sklearn.preprocessing import OneHotEncoder

# Read in data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "csv_data", "sb.csv")
sbdf = pd.read_csv(DATA_PATH)

# Clean data
sbdf_clean = sbdf

# Null count summed across all columns
numNulls = sbdf_clean.isnull().sum().sum()
if numNulls > 0:
    sbdf_clean = sbdf_clean.dropna()

# Dup-row count summed
numDups = sbdf_clean.duplicated().sum()
if numDups > 0:
    sbdf_clean = sbdf_clean.drop_duplicates()

# Finding and removing outliers (remove row if an attribute in any numeric column is an outlier)
valid_rows = pd.Series(True, index=sbdf_clean.index)
for col_name, col_data in sbdf_clean.select_dtypes(include=["int64", "float64"]).items():
    z_scores = stats.zscore(col_data)
    outliers = abs(z_scores) > 3
    
    if outliers.any():
        valid_rows = valid_rows & ~outliers # essentially a boolean expression

sbdf_clean = sbdf_clean[valid_rows].reset_index(drop=True)

CLEANED_DF_PATH = os.path.join(SCRIPT_DIR, "..", "csv_data", "sb_clean.csv")
sbdf_clean.to_csv(CLEANED_DF_PATH, index=False)

# Generate vector embeddings

# Milvus article on embeddings https://medium.com/vector-database/how-to-get-the-right-vector-embeddings-83295ced7f35
# Milvus article on vector similariy search https://zilliz.com/learn/vector-similarity-search

# For now:
#   Drop id col
#   Normalizing all continuous numeric columns
#   Converting all two-category columns to be 0-or-1
#   Converting all categorical columns to be one-hot-encoded
# This should give us a "vector embedding", albeit a very simple one

normalization_cols = [
    "Annual_Revenue",
    "Debt_To_Income_Ratio",
    "Credit_Score",
    "Loan_Amount_Requested",
    "Loan_Term_Months",
    "Interest_Rate",
    "Past_Loan_Defaults",
]

two_category_cols = {
    "Approval_Status": {
        "Approved": 1,
        "Denied": 0
    }
}

one_hot_cols = [
    "Business_Category"
]

sbdf_ve = sbdf_clean.drop("Loan_ID", axis=1)
for col_name in sbdf_ve.columns:
    
    if col_name in normalization_cols:
        sbdf_ve[col_name] = stats.zscore(sbdf_ve[col_name])
    
    if col_name in two_category_cols.keys():
        sbdf_ve[col_name] = sbdf_ve[col_name].map(
            two_category_cols[col_name]
        )

    if col_name in one_hot_cols:
        col_encoder = OneHotEncoder(sparse_output=False)
        col_encoded = col_encoder.fit_transform(sbdf_ve[[col_name]])
        df_encoded = pd.DataFrame(
            col_encoded,
            columns=col_encoder.get_feature_names_out([col_name])
        )
        sbdf_ve = pd.concat([sbdf_ve, df_encoded], axis=1)
        sbdf_ve = sbdf_ve.drop(col_name, axis=1)


VE_DF_PATH = os.path.join(SCRIPT_DIR, "..", "csv_data", "sb_ve.csv")
sbdf_ve.to_csv(VE_DF_PATH, index=False)