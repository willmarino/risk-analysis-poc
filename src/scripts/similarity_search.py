import time
from ..services.zilliz import fetch_vectors, single_vector_search

# Zilliz docs were super helpful
# https://docs.zilliz.com/docs/quick-start#similarity-search

# Pulling some vectors from the validation set, and using them as queries to search the train set
# This is how I am simulating a "new applicant"
print("Running similarity search for 5 vectors...")

NUM_RUNS = 5
for idx in range(0, NUM_RUNS):

    # Fetch sample vector
    validation_data = fetch_vectors("sbl_val", idx, 1)
    validation_vector = validation_data[0]["vector"]

    start_time = time.time()

    # Run single vector similarity search
    search_response = single_vector_search("sbl_train", validation_vector)

    # Sort response, grab min value
    search_response_sorted = sorted(search_response, key=lambda x: x["distance"])
    closest_vector = search_response_sorted[0]

    end_time = time.time()
    time_diff = end_time - start_time

    print(f"Found closest vector in {time_diff:.2f} seconds")