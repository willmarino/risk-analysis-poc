from ..services.zilliz import single_vector_search

# Zilliz docs were super helpful
# https://docs.zilliz.com/docs/quick-start#similarity-search

# Just took the VE which was autofilled for this route from Zilliz's API playground
dummy_vector_embedding = [
    0.26651621043836027,
    0.963506742923351,
    0.06656141746585242,
    0.0791575852228148,
    0.2470072780022039,
    0.09536912435302902,
    0.6287248766271651,
    0.3517254453998986,
    0.7332961863276148,
    0.0819152543771533,
    0.5999939697072323,
    0.027695362670982604,
    0.5887723007321608
]

response = single_vector_search("sbl_train", dummy_vector_embedding)

print(response)