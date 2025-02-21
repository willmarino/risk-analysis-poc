import os
import json
import requests
from dotenv import load_dotenv

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(CUR_PATH, "..", "..", ".env.dev")
load_dotenv(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        ".env.dev"
    )
)

def get_base_url():
    return f"{os.getenv("ZILLIZ_CLUSTER_ENDPOINT")}/v2/vectordb/entities"

def get_headers():
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {os.getenv("ZILLIZ_BEARER_TOKEN")}",
    }

def insert_embeddings(col_name, embeddings, status):
    url = f"{get_base_url()}/insert"
    
    formatted_embeddings = []
    for i in range(len(embeddings)):
        formatted_embeddings.append({"vector": embeddings[i], "status": status[i]})
    
    payload = {
        "collectionName": col_name,
        "data": formatted_embeddings
    }

    response = requests.post(
        url,
        data=json.dumps(payload),
        headers=get_headers()
    )

    insertion_ids = response.json()["data"]["insertIds"]
    return insertion_ids


def fetch_vectors(col_name, offset, limit):
    url = f"{get_base_url()}/query"

    payload = {
        "collectionName": col_name,
        "outputFields": ["vector", "status"],
        "offset": offset,
        "limit": limit
    }

    response = requests.post(
        url,
        data=json.dumps(payload),
        headers=get_headers()
    )

    return response.json()["data"]


def single_vector_search(col_name, v_e):
    url = f"{get_base_url()}/search"

    payload = {
        "collectionName": col_name,
        "data": [v_e],
        "outputFields": [
            "*"
        ]
    }

    response = requests.post(
        url,
        data=json.dumps(payload),
        headers=get_headers()
    )

    return response.json()["data"]
