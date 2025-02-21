import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from ..services.zilliz import fetch_vectors

# from ..services.random_forest import rf, new_rf, train_inputs, train_outputs
from ..services.random_forest import get_model
from ..services.open_ai import generate_explanation

# Fetch random forest model
rf = get_model()

# Model is already trained in ..models.random_forest,
# just doing prediction testing on validation data (val_ve) here
val_ve = fetch_vectors("sbl_val", 0, 1000)
val_inputs = pd.DataFrame(list(map(lambda x: x["vector"], val_ve)))
val_outputs = list(map(lambda x: x["status"], val_ve))

val_predictions = rf.predict(val_inputs)
accuracy = accuracy_score(val_outputs, val_predictions)

print(f"Ran predictions on validation data, accuracy score: {accuracy}")

# Demoing the use of chatgpt for explaining ouput values for randomly selected data
[random_int] = np.random.randint(0, len(val_outputs), size=1)
random_input = val_inputs.iloc[random_int]
random_output = val_outputs[random_int]

# message = generate_explanation(rf, train_inputs, train_outputs, random_input, random_output)
message = generate_explanation(random_input, random_output)
print(message)