import pandas as pd

from ..services.zilliz import insert_embeddings, describe_collection
from ..services.df_util import read_df_from_csv, write_df_to_csv, gen_train_val_split, clean_df, gen_vector_embeddings

# Read initial csv data
sbdf = read_df_from_csv("sb.csv")

# Generate cleaned df
sbdf_clean = clean_df(sbdf)

# Write cleaned df to local dir - for debugging
write_df_to_csv(sbdf_clean, "sb_clean.csv")

# Generate a df filled with vector embeddings (all numerical data, normalized where needed)
df_ve = gen_vector_embeddings(sbdf_clean)

# Separate prepared dataframe out into training and validation sets (will be helpful for sim search and building model)
train_ve, val_ve = gen_train_val_split(df_ve)

train_col_response = describe_collection("sbl_train")
if len(train_col_response["message"]) > 0: # Could not find collection
    raise Exception("Could not find sbl_train cluster, please create it following the instructions in README.md")

val_col_response = describe_collection("sbl_val")
if len(val_col_response["message"]) > 0: # Could not find collection
    raise Exception("Could not find sbl_val cluster, please create it following the instructions in README.md")

# Write embeddings in Zilliz, only include relevant data
train_insertion_ids = insert_embeddings(
    "sbl_train",
    train_ve.drop(columns=["Loan_ID", "Approval_Status"]).values.tolist(),
    train_ve["Approval_Status"].values.tolist()
)

train_ve = pd.concat(
    [train_ve, pd.Series(train_insertion_ids, name="zilliz_insertion_id")],
    axis=1
)

write_df_to_csv(train_ve, "train_ve.csv")

# Same with validation set
val_insertion_ids = insert_embeddings(
    "sbl_val",
    val_ve.drop(columns=["Loan_ID", "Approval_Status"]).values.tolist(),
    val_ve["Approval_Status"].values.tolist()
)

val_ve = pd.concat(
    [val_ve, pd.Series(val_insertion_ids, name="zilliz_insertion_id")],
    axis=1
)

write_df_to_csv(val_ve, "val_ve.csv")
