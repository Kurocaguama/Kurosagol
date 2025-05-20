from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_sim(s1, s2):
    emb1 = model.encode(s1)
    emb2 = model.encode(s2)

    sims = model.similarity(emb1, emb2)
    return round(sims[0][0].item(), 3)