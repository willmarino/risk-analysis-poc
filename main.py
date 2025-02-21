import time
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List

from src.services.zilliz import single_vector_search
from src.services.open_ai import generate_explanation
from services.random_forest import rf


app = FastAPI()

@app.middleware("http")
async def add_timing_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    print(f"Request to {request.url.path} took {process_time:.3f} seconds")
    return response


@app.get("/")
def ping():
    return {"bing": "bong"}


class VectorEmbedding(BaseModel):
    vector: List[float]

@app.post("/similarity_retrieval")
def similarity_retrieval(vector_embedding: VectorEmbedding):
    search_response = single_vector_search("sbl_train", vector_embedding.vector)
    
    # should encapsulate this somewhere, is dupped from similarity_search.py
    search_response_sorted = sorted(search_response, key=lambda x: x["distance"])
    closest_vector = search_response_sorted[0]

    return {
        "closest_vector": closest_vector
    }

@app.post("/risk_scoring")
def risk_scoring(vector_embedding: VectorEmbedding):
    [prediction] = rf.predict([vector_embedding.vector])
    explanation = generate_explanation(vector_embedding.vector, prediction)

    return {
        "predicted_status": "Approved" if prediction == 1 else "Denied",
        "explanation": explanation
    }

