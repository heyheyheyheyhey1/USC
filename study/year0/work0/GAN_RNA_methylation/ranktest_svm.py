import os
import pandas as pd
import torch
from WGANGP import Discriminator
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.svm import SVC
from tqdm import tqdm
import pickle

DATA_DIR = os.path.join("data")
MODEL_DIR = os.path.join("model", "SVM")

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
mdls = os.listdir(MODEL_DIR)
indexes = [f'round_{i}.sav' for i in range(len(mdls))]
columns = [f'gene_{i}' for i in range(gene_nums)]

average_pred = pd.DataFrame(index=indexes, columns=columns)
for mdl in tqdm(mdls):
    svc = pickle.load(open(os.path.join(MODEL_DIR, mdl), 'rb'))
    pred = svc.predict(x)
    average_pred.loc[mdl] = pred

average_pred.loc["mean"] = average_pred.mean()

pred = average_pred.loc["mean"].values
fpr, tpr, _ = roc_curve(y, pred, drop_intermediate=False)
roc_auc_score = auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.05])
# plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('WGAN + SVM ROC CURVE')
plt.legend(loc=4)
plt.grid(True)
plt.tight_layout()
plt.savefig('ROC_WGAN_SVM.png')
plt.show()
