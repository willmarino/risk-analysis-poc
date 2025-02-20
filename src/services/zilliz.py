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

def getBaseUrl():
    return f"{os.getenv("ZILLIZ_CLUSTER_ENDPOINT")}/v2/vectordb/entities"

def getHeaders():
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {os.getenv("ZILLIZ_BEARER_TOKEN")}",
    }

def insertEmbeddings(embeddings):
    url = f"{getBaseUrl()}/insert"
    
    formatted_embeddings = []
    for e in embeddings:
        formatted_embeddings.append({"vector": e})
    
    payload = {
        "collectionName": "sbl",
        "data": formatted_embeddings
    }

    response = requests.post(
        url,
        data=json.dumps(payload),
        headers=getHeaders()
    )

    insertion_ids = response.json()["data"]["insertIds"]
    return insertion_ids
