import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from ..services.zilliz import fetch_vectors

# Load train and val dataframes from zilliz
train_ve = fetch_vectors("sbl_train", 0, 1000)
train_inputs = pd.DataFrame(list(map(lambda x: x["vector"], train_ve)))
train_outputs = list(map(lambda x: x["status"], train_ve))


val_ve = fetch_vectors("sbl_val", 0, 1000)
val_inputs = pd.DataFrame(list(map(lambda x: x["vector"], val_ve)))
val_outputs = list(map(lambda x: x["status"], val_ve))

# Train random forest, evaluate accuracy

rf = RandomForestClassifier()

rf.fit(train_inputs.values, train_outputs)

val_predictions = rf.predict(val_inputs)

accuracy = accuracy_score(val_outputs, val_predictions)

print(f"Accuracy score: {accuracy}")








