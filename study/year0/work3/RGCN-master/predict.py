import torch
import pandas as pd
import numpy as np
import os
from models import RGCN
from utils import *

DATA_DIR = os.path.join("data", "drug_disease_drug")

entities_dict = pd.read_csv(os.path.join(DATA_DIR, "entities.dict"), sep='\t', header=None).to_dict(orient='series')[1]
relation_dict = pd.read_csv(os.path.join(DATA_DIR, "relations.dict"), sep='\t', header=None).to_dict(orient='series')[1]
# all_triplets = read_triplets(os.path.join(DATA_DIR, "train.txt"),entities_dict,relation_dict)
all_triplet = load_data('./data/drug_disease_drug')[2]
pass


def load_model(filepath=None):
    checkpoint = torch.load(filepath)
    m = RGCN(1105, 1283, 4, 1)
    m.load_state_dict(checkpoint['state_dict'])
    return m


def gen_predict_tri(n=100, t=1):
    trip = []
    for i in range(n):
        for j in range(i + 1, n):
            tri = [i, t, j]
            trip.append(tri)
    trip = pd.DataFrame(trip).to_numpy()
    return trip



model = load_model(os.path.join('best_mrr_model.pth'))
model.eval()

predict_tri = gen_predict_tri(1105,1)
predict_data = build_test_graph(len(entities_dict), len(relation_dict), all_triplet)
entity_embedding = model(predict_data.entity, predict_data.edge_index, predict_data.edge_type, predict_data.edge_norm)
score = model.distmult(entity_embedding, predict_tri)  # 1105x100 4950*3

_, idx = torch.topk(score , 10)
predict_tri[idx]

pass
