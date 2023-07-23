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

CLASSIFIERS_NAMES = ['SVM', 'GB', 'GNB', 'LR', 'RF']
GAN_SCORE = pd.read_csv(os.path.join("GAN_SCORE.CSV"),sep="\t")
SMOTE_SCORE = pd.read_csv(os.path.join("smote_SCORE.CSV"),sep="\t")
Y_true = GAN_SCORE.loc[:,"y_gan_true"]


def sorted_score(a,b):
    sorted_pairs = sorted(zip(a.copy(), b.copy()), reverse=True)
    a, b = zip(*sorted_pairs)
    return a,b

def plot_auc_pr(y_true,y_prob_gan,y_prob_smote,classifier):
    y_score_gan ,y_true_gan = sorted_score(y_prob_gan,y_true)
    y_score_smote ,y_true_smote = sorted_score(y_prob_smote,y_true)


    fpr_gan, tpr_gan, _ = roc_curve(y_true_gan, y_score_gan, drop_intermediate=False)
    area_roc_gan = auc(fpr_gan, tpr_gan)

    precision_gan, recall_gan, _ = precision_recall_curve(y_true_gan, y_score_gan)
    area_pr_gan = auc(recall_gan, precision_gan)

    fpr_smote, tpr_smote, _ = roc_curve(y_true_smote, y_score_smote, drop_intermediate=False)
    area_roc_smote = auc(fpr_smote, tpr_smote)

    precision_smote, recall_smote, _ = precision_recall_curve(y_true_smote, y_score_smote)
    area_pr_smote = auc(recall_smote, precision_smote)

    # plt.figure(figsize=(20,5))
    fig, (p1, p2) = plt.subplots(1,2,figsize=(10, 6))
    p1.plot(fpr_gan, tpr_gan, color='blue', lw=1,label = f'WGAN area = {area_roc_gan:.4f}')
    p1.plot(fpr_smote, tpr_smote, color='red', lw=1,label = f'SMOTE area = {area_roc_smote:.4f}')
    p1.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', alpha=0.4)
    p1.set_xlabel('False Positive Rate')
    p1.set_ylabel('True Positive Rate')
    p1.set_title(f'ROC CURVE')
    p1.legend()
    p1.grid(True)

    p2.plot(recall_gan, precision_gan, color='blue', lw=1,label = f'WGAN area = {area_pr_gan:.4f}')
    p2.plot(recall_smote, precision_smote, color='red', lw=1,label = f'SMOTE area = {area_pr_smote:.4f}')
    p2.set_xlabel('Recall', )
    p2.set_ylabel('Precision')
    p2.plot([0, 1], [1, 0], color='navy', lw=1, linestyle='--', alpha=0.4)
    p2.set_title('PR CURVE')
    p2.legend()
    p2.grid(True)

    p1.set_aspect('equal')
    p2.set_aspect('equal')
    plt.tight_layout()
    plt.suptitle(f"{classifier_name} CURVES")
    plt.savefig(f'plot_smote_gan_{classifier_name}.png')
    plt.show()


for classifier_name in tqdm(CLASSIFIERS_NAMES):
    score_gan = GAN_SCORE.loc[:,f"{classifier_name}_GAN_score"]
    score_smote = SMOTE_SCORE.loc[:,f"{classifier_name}_smote_score"]
    plot_auc_pr(Y_true,score_gan,score_smote,classifier_name)

