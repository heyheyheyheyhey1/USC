import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *
from random import shuffle
from tqdm import tqdm
import pickle
from datetime import datetime

DATA_DIR = os.path.join("data")
MODEL_DIR = os.path.join("model")
OUT_DIR = os.path.join("result", datetime.now().strftime("%Y-%m-%d %Hh%Mm"))
CLASSIFIER_NAMES = ['GB', 'GNB', 'LR', 'RF', 'SVM']

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

selected_data = "selected_dataset.tsv"
dataset_matrix = os.path.join(DATA_DIR, selected_data)
selected_data = pd.read_csv(dataset_matrix, delimiter="\t", index_col=0, low_memory=False)
positive_genes = [line.rstrip('\n') for line in open(os.path.join(DATA_DIR, "test_positive_genes.txt"))]
negative_genes = [line.rstrip('\n') for line in open(os.path.join(DATA_DIR, "test_negative_genes.txt"))]
positive_data = selected_data.loc[positive_genes].values
negative_data = selected_data.loc[negative_genes].values
shuffle(negative_data)

average_pred = pd.DataFrame(columns=['model', 'batch_n', 'Accuracy', "Precision", "Recall", "F1"])
pd_results = pd.DataFrame(columns=['model', 'Accuracy', "Precision", "Recall", "F1"])


def data_enumerator():
    postv_num = len(positive_data)
    negtv_num = len(negative_data)
    for i in range(int(negtv_num / postv_num)):
        neg_batch = negative_data[i * postv_num:(i + 1) * postv_num]
        x = np.concatenate([np.array(neg_batch), positive_data])
        y0 = np.zeros([len(neg_batch), ])
        y1 = np.ones([postv_num, ])
        y = np.concatenate([y0, y1], axis=0)
        yield x, y, i


for classifier in tqdm(CLASSIFIER_NAMES):
    for mdl in os.listdir(os.path.join(MODEL_DIR, classifier)):
        for x, y, i in data_enumerator():
            model = pickle.load(open(os.path.join(MODEL_DIR, classifier, mdl), 'rb'))
            pred = model.predict(x)
            scorings = [classifier, f"batch_{i}", accuracy_score(y, pred), precision_score(y, pred),
                        recall_score(y, pred),
                        f1_score(y, pred)]
            average_pred.loc[len(average_pred)] = scorings

average_pred.round(4)

for classifier in tqdm(CLASSIFIER_NAMES):
    test_pd = average_pred[average_pred["model"] == classifier].copy()
    mean = test_pd.mean(numeric_only=True, skipna=True)
    mean.loc["model"] = classifier
    pd_results.loc[len(pd_results)] = mean
    test_pd.to_csv(os.path.join(OUT_DIR, f"{classifier}_GAN_test.csv"), index=True, sep="\t")

pd_results.loc[len(pd_results)] = pd_results.mean()
pd_results.to_csv(os.path.join(OUT_DIR, f"GAN_test.csv"), sep="\t", index=False)

# average_pred.loc["mean"] = average_pred.mean(numeric_only=True)
# average_pred.to_csv(os.path.join(OUT_DIR, "SVM_GAN_test.csv"), index=True)

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
