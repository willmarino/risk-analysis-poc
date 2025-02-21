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
    return f"{os.getenv("ZILLIZ_CLUSTER_ENDPOINT")}/v2/vectordb"

def get_headers():
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {os.getenv("ZILLIZ_BEARER_TOKEN")}",
    }

def describe_collection(col_name):
    print(f"Checking to see if collection exists on zilliz: {col_name}...")
    url = f"{get_base_url()}/collections/describe"
    payload = { "collectionName": col_name }

    response = requests.post(
        url,
        data=json.dumps(payload),
        headers=get_headers()
    )

    return response.json()


# def create_collection(col_name):
#     print(f"Creating new collection on zilliz: {col_name}...")
#     url = f"{get_base_url()}/collections/create"
#     payload = {
#         "collection_name": col_name,
#         "dimensions": 13,
#         "metricType": "COSINE",
#         "vectorField": "vector"
#     }


def insert_embeddings(col_name, embeddings, status):
    print(f"Inserting vector embeddings into zilliz: {col_name}...")
    url = f"{get_base_url()}/entities/insert"
    
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
    print(f"Fetching vector embeddings from zilliz: {col_name}...")
    url = f"{get_base_url()}/entities/query"

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
    print(f"Running vector search on zilliz: {col_name}...")
    url = f"{get_base_url()}/entities/search"

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
