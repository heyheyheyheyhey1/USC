import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *

from tqdm import tqdm
import pickle

DATA_DIR = os.path.join("data")
MODEL_DIR = os.path.join("model", "SVM")
OUT_DIR = os.path.join("result")

selected_data = "selected_dataset.tsv"
dataset_matrix = os.path.join(DATA_DIR, selected_data)
selected_data = pd.read_csv(dataset_matrix, delimiter="\t", index_col=0, low_memory=False)
positive_genes = [line.rstrip('\n') for line in open(os.path.join(DATA_DIR, "test_positive_genes.txt"))]
negative_genes = [line.rstrip('\n') for line in open(os.path.join(DATA_DIR, "test_negative_genes.txt"))]
positive_data = selected_data.loc[positive_genes].values
negative_data = selected_data.loc[negative_genes].values

mdls = os.listdir(MODEL_DIR)
# indexes = [f'round_{i}.sav' for i in range(len(mdls))]
columns = ['model','batch_n', 'Accuracy', "Precision", "Recall", "F1"]
average_pred = pd.DataFrame(columns=columns)


def data_enumerator():
    postv_num = len(positive_data)
    for i in range(int(len(negative_data) / len(positive_data))):
        neg_batch = positive_data[i * postv_num:(i + 1) * postv_num]
        x = np.concatenate([np.array(neg_batch), positive_data])
        y0 = np.zeros([len(neg_batch), ])
        y1 = np.ones([postv_num, ])
        y = np.concatenate([y0, y1], axis=0)
        yield x, y, i


for mdl in tqdm(mdls):
    svc = pickle.load(open(os.path.join(MODEL_DIR, mdl), 'rb'))
    for x, y, i in data_enumerator():
        pred = svc.predict(x)
        scorings = [mdl,f"batch_{i}", accuracy_score(y, pred), precision_score(y, pred), recall_score(y, pred),
                    f1_score(y, pred)]
        average_pred.loc[len(average_pred)] = scorings

average_pred.loc["mean"] = average_pred.mean(numeric_only=True)
average_pred.to_csv(os.path.join(OUT_DIR, "SVM_SMOTE_test.csv"), index=True)

# fpr, tpr, _ = roc_curve(y, pred, drop_intermediate=False)
# roc_auc_score = auc(fpr, tpr)
#
# plt.figure()
#
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc_score)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# # plt.xlim([0.0, 1.05])
# # plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('WGAN + SVM ROC CURVE')
# plt.legend(loc=4)
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('ROC_WGAN_SVM.png')
# plt.show()
