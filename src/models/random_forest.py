import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from ..services.zilliz import fetch_vectors

# Load train and val dataframes from zilliz
train_ve = fetch_vectors("sbl_train", 0, 1000)
train_inputs = pd.DataFrame(list(map(lambda x: x["vector"], train_ve)))
train_outputs = list(map(lambda x: x["status"], train_ve))

rf = RandomForestClassifier()

rf.fit(train_inputs.values, train_outputs)