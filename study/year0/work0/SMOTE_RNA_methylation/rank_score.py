import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.svm import SVC
from tqdm import tqdm
import pickle

DATA_DIR = os.path.join("data")
MODEL_DIR = os.path.join("model")
CLASSIFIERS_NAMES = ['SVM', 'GB', 'GNB', 'LR', 'RF']
selected_data = "selected_dataset.tsv"
dataset_matrix = os.path.join(DATA_DIR, selected_data)
selected_data = pd.read_csv(dataset_matrix, delimiter="\t", index_col=0, low_memory=False)
positive_genes = [line.rstrip('\n') for line in open(os.path.join(DATA_DIR, "test_positive_genes.txt"))]
negative_genes = [line.rstrip('\n') for line in open(os.path.join(DATA_DIR, "test_negative_genes.txt"))]
positive_data = selected_data.loc[positive_genes].values
negative_data = selected_data.loc[negative_genes].values

y0 = np.zeros([len(negative_data), ])
y1 = np.ones([len(positive_data), ])
# 定义y
y = np.concatenate([y0, y1], axis=0)
# 定义x
x = np.concatenate([np.array(negative_data), positive_data])
x = torch.tensor(x, dtype=torch.float32)
# 打乱
# x,y = shuffle(x,y)
gene_nums = len(x)
indexes = [f'gene_{i}' for i in range(gene_nums)]
pd_out = pd.DataFrame()
pd_out.insert(column="y_smote_true", value=y, loc=0)
for classifier in tqdm(CLASSIFIERS_NAMES):
    mdls = os.listdir(os.path.join(MODEL_DIR, classifier))
    average_pred = pd.DataFrame()
    for i, mdl in tqdm(enumerate(mdls), total=len(mdls)):
        model = pickle.load(open(os.path.join(MODEL_DIR, classifier, mdl), 'rb'))
        pred = model.predict_proba(x)[:, -1]
        average_pred.insert(value=pred, column=f'round_{i}', loc=len(average_pred.columns))
    mean = average_pred.mean(axis=1)
    pd_out.insert(column=f'{classifier}_smote_score', value=mean, loc=len(pd_out.columns))

pd_out.to_csv(os.path.join("smote_SCORE.CSV"),index=False,sep='\t')