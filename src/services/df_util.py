import os
import pandas as pd
from scipy import stats
from sklearn.preprocessing import OneHotEncoder

# This file is used for functions related to data ingestion and cleaning


def read_df_from_csv(csv_file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    read_path = os.path.join(script_dir, "..", "..", "csv_data", csv_file_name)
    df = pd.read_csv(read_path)
    return df


def write_df_to_csv(df, csv_file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    write_path = os.path.join(script_dir, "..", "..", "csv_data", csv_file_name)
    df.to_csv(write_path, index=False)


# Should I be reseting index for val df?
def gen_train_val_split(df):
    split_point = int(len(df) * 0.8) # 80% train, 20% val, probs should randomize but want to keep this simple rn
    train = df[:split_point]
    val = df[split_point:].reset_index(drop=True)

    return [train, val]


def clean_df(df):
    # Null count summed across all columns
    numNulls = df.isnull().sum().sum()
    if numNulls > 0:
        df = df.dropna()

    # Dup-row count summed
    numDups = df.duplicated().sum()
    if numDups > 0:
        df = df.drop_duplicates()

    # Finding and removing outliers (remove row if an attribute in any numeric column is an outlier)
    valid_rows = pd.Series(True, index=df.index)
    for col_name, col_data in df.select_dtypes(include=["int64", "float64"]).items():
        z_scores = stats.zscore(col_data)
        outliers = abs(z_scores) > 3
        
        if outliers.any():
            valid_rows = valid_rows & ~outliers # essentially a boolean expression

    df = df[valid_rows].reset_index(drop=True)

    return df


# Generate vector embeddings
# Milvus article on embeddings https://medium.com/vector-database/how-to-get-the-right-vector-embeddings-83295ced7f35
# Milvus article on vector similariy search https://zilliz.com/learn/vector-similarity-search

# For now:
#   Drop id col
#   Normalizing all continuous numeric columns
#   Converting all two-category columns to be 0-or-1
#   Converting all categorical columns to be one-hot-encoded
# This should give us a "vector embedding", albeit a very simple one
def gen_vector_embeddings(df):

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

    df_ve = df
    for col_name in df_ve.columns:
        
        if col_name in normalization_cols:
            df_ve[col_name] = stats.zscore(df_ve[col_name])
        
        if col_name in two_category_cols.keys():
            df_ve[col_name] = df_ve[col_name].map(
                two_category_cols[col_name]
            )

        if col_name in one_hot_cols:
            col_encoder = OneHotEncoder(sparse_output=False)
            col_encoded = col_encoder.fit_transform(df_ve[[col_name]])
            df_encoded = pd.DataFrame(
                col_encoded,
                columns=col_encoder.get_feature_names_out([col_name])
            )
            df_ve = pd.concat([df_ve, df_encoded], axis=1)
            df_ve = df_ve.drop(col_name, axis=1)
    
    return df_ve