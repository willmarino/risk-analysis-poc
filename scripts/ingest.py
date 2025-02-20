import os
import pandas as pd
from scipy import stats

# Read in data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "sample_data", "ssbld.csv")
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
for col_name, col_data in sbdf_clean.select_dtypes(include=['int64', 'float64']).items():
    z_scores = stats.zscore(col_data)
    outliers = abs(z_scores) > 3
    
    if outliers.any():
        valid_rows = valid_rows & ~outliers # essentially a boolean expression

sbdf_clean = sbdf_clean[valid_rows]

# Generate vector embeddings