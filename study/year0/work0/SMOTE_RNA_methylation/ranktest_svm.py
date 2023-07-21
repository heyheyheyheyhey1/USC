import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from sklearn.metrics import roc_curve, auc, roc_auc_score,precision_recall_curve
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
indexes = [f'round_{i}.sav' for i in range(176)]
columns = [f'gene_{i}' for i in range(gene_nums)]

average_pred = pd.DataFrame(index=indexes, columns=columns)
for mdl in tqdm(os.listdir(MODEL_DIR)):
    svc = pickle.load(open(os.path.join(MODEL_DIR, mdl), 'rb'))
    pred = svc.predict(x)
    average_pred.loc[mdl] = pred

average_pred.loc["mean"] = average_pred.mean()
proba_mean = average_pred.loc["mean"].values
sorted_pairs = sorted(zip(proba_mean, y), reverse=True)
proba_mean, y = zip(*sorted_pairs)

fpr, tpr, _ = roc_curve(y, proba_mean, drop_intermediate=False)
roc_auc_score = auc(fpr, tpr)

precision, recall, thresholds = precision_recall_curve(y, proba_mean)
area_pr = auc(recall, precision)


fig, (p1, p2) = plt.subplots(1, 2)
p1.plot(fpr, tpr, color='darkorange', lw=2)
p1.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', alpha=0.4)
p1.set_xlabel('False Positive Rate')
p1.set_ylabel('True Positive Rate')
p1.set_title(f'ROC CURVE (AUC = {roc_auc_score:.4f})')
p1.grid(True)

p2.plot(recall, precision, color='darkorange')
p2.set_xlabel('Recall', fontsize=12)
p2.set_ylabel('Precision', fontsize=12)
p2.plot([0, 1], [1, 0], color='navy', lw=1, linestyle='--', alpha=0.4)
p2.set_title('PR Curve (AUC = %0.4f)' % area_pr)
p2.grid(True)

p1.set_aspect('equal')
p2.set_aspect('equal')

plt.tight_layout()
plt.suptitle("SMOTE + SVM CURVES")
plt.savefig('ROC_SMOTE_SVM.png')
plt.show()
