from sklearn.metrics import *
import pandas as pd
import numpy as np


def precision_k(y_true, y_score, threshold=0.5, k=10):
    sorted_pairs = sorted(zip(y_score, y_true), reverse=True)
    y_proba, y_true = zip(*sorted_pairs)
    y_proba_fix = [1 if a>threshold else 0 for a in y_proba]
    y_proba_k = y_proba_fix[:k]
    y_true_k = y_true[:k]
    TP = sum([1 if a == b and b == 1 else 0 for a, b in zip(y_proba_k, y_true_k)])
    return TP/k


def recall_k(y_true, y_score, threshold=0.5, k=10):
    sorted_pairs = sorted(zip(y_score, y_true), reverse=True)
    y_proba, y_true = zip(*sorted_pairs)
    y_proba_fix = [1 if a>threshold else 0 for a in y_proba]
    score = recall_score(y_true[:k], y_proba_fix[:k])
    return score


def f1_k(y_true, y_score, threshold=0.5, k=10):
    sorted_pairs = sorted(zip(y_score, y_true), reverse=True)
    y_proba, y_true = zip(*sorted_pairs)
    y_proba_fix = [1 if a>threshold else 0 for a in y_proba]
    score = f1_score(y_true[:k], y_proba_fix[:k])
    return score
