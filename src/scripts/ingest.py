import pandas as pd

from ..services.zilliz import insert_embeddings
from ..services.df_util import read_df_from_csv, write_df_to_csv, gen_train_val_split, clean_df, gen_vector_embeddings

# Read initial csv data
sbdf = read_df_from_csv("sb.csv")

# Generate cleaned df
sbdf_clean = clean_df(sbdf)

# Write cleaned df to local dir
write_df_to_csv(sbdf_clean, "sb_clean.csv")

# Generate a df filled with vector embeddings (all numerical data, normalized where needed)
df_ve = gen_vector_embeddings(sbdf_clean)

# Separate prepared dataframe out into training and validation sets (will be helpful for sim search and building model)
train_ve, val_ve = gen_train_val_split(df_ve)

# Write embeddings in Zilliz, only include relevant data
# Adding insertion_ids to dataframes
# Doing this because we don't want to store approval status in zilliz,
# but we will need to know which vector embeddings represent approvals or denials
# in order to train a model to make predictions
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
