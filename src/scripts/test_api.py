import requests

response = requests.post(
    "http://localhost:8000/similarity_retrieval",
    json={
        "vector": [1.2254771,-1.5213871,0.856086,-0.9040651,0.6656052,0.37095428,2.7211514,1,0,0,0,0,0],
    }
)
print(response.json())

# response = requests.post(
#     "http://localhost:8000/risk_scoring",
#     json={
#         "vector": [1.2254771,-1.5213871,0.856086,-0.9040651,0.6656052,0.37095428,2.7211514,1,0,0,0,0,0],
#     }
# )
# print(response.json())