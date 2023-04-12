import random
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
import pandas as pd
import pickle
import os
from random import sample
from sklearn.utils import shuffle
import tqdm
from imblearn.over_sampling import *
from sklearn.model_selection import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import *
import numpy as np
CV_DIR = os.path.join("CV", "AL")

MODEL_DIR = os.path.join("model","AL")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(CV_DIR):
    os.makedirs(CV_DIR)

selected_data = "selected_dataset.tsv"
rnmts = "RNMT.list"
dataset_matrix = os.path.join("data", selected_data)
selected_data = pd.read_csv(dataset_matrix, delimiter="\t", index_col=0, low_memory=False)

all_genes = selected_data.index.unique().to_list()
positive_genes = [line.rstrip('\n') for line in open(os.path.join("data", rnmts))]
negative_genes = set(all_genes).difference(positive_genes)  # 负样本

selected_data["Y"] = [1 if idx in positive_genes else 0 for idx in selected_data.index.to_list()]
test_positive_genes = sample(positive_genes, int(0.2 * len(positive_genes)))  # 取正样本20%作为测试
test_negative_genes = sample(negative_genes, int(0.2 * len(negative_genes)))  # 测试负样本
train_positive_genes = set(positive_genes).difference(test_positive_genes)
train_negative_genes = set(negative_genes).difference(test_negative_genes)
train_positive_frame = selected_data.loc[list(train_positive_genes)]
train_negative_frame = selected_data.loc[list(train_negative_genes)]

print("all genes num: %d\n" % (len(all_genes)))
print("test genes num: %d\n" % (len(test_negative_genes) + len(test_positive_genes)))

random.seed(42)

# print("positive num before oversample: %d" % (len(train_y)))
# train_x_os, train_y_os = SMOTEN(random_state=42, sampling_strategy={1: 2000}).fit_resample(train_x, train_y)
# print("length after oversample: %d" % (len(train_x_os)))

# shuffle_idx = shuffle(range(len(train_x_os)))
# train_x_os = train_x_os[shuffle_idx]
# train_y_os = train_y_os[shuffle_idx]


def mod_tunning(n, X_train, y_train):
    query_num = 11
    n_initial = 20
    initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
    learner = ActiveLearner(
        estimator=GradientBoostingClassifier(),
        query_strategy=uncertainty_sampling,
        X_training=X_train[initial_idx], y_training=y_train[initial_idx]
    )
    for i in range(query_num):
        query_idx, query_result = learner.query(X_train)
        learner.teach(X_train[query_idx], y_train[query_idx])
    return learner


cv_scorings = pd.DataFrame(columns=('Round', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'))


def train_one_epoch(X, Y, n):
    AL_model = mod_tunning(n, X, Y)
    scorings = {'accuracy': make_scorer(accuracy_score),
                'recall': make_scorer(recall_score),
                'precision': make_scorer(precision_score),
                'f1_score': make_scorer(f1_score),
                'auc': 'roc_auc'}

    scores = cross_validate(estimator=AL_model.estimator, X=X, y=Y, cv=10, scoring=scorings, n_jobs=12)
    pickle.dump(AL_model, open(os.path.join(MODEL_DIR, f'round_{n}.sav'), 'wb'))
    classifier_performance = {'Round': n,
                              'Accuracy': f'{scores["test_accuracy"].mean():.5f}',
                              'Precision': f'{scores["test_precision"].mean():.5f}',
                              'Recall': f'{scores["test_recall"].mean():.5f}',
                              'F1': f'{scores["test_f1_score"].mean():.5f} ',
                              'AUC': f'{scores["test_auc"].mean():.5f}'}
    cv_scorings.loc[len(cv_scorings)] = classifier_performance

    if (n % 3 == 0):
        print(f'\naccuracy:{scores["test_accuracy"].mean():.5f} (+/- {(scores["test_accuracy"].std() * 2):.5f})')
        print(f'Precision:{scores["test_precision"].mean():.5f} (+/- {(scores["test_precision"].std() * 2):.5f})')
        print(f'Recall:{scores["test_recall"].mean():.5f} (+/- {(scores["test_recall"].std() * 2):.5f})')
        print(f'F1:{scores["test_f1_score"].mean():.5f} (+/- {(scores["test_f1_score"].std() * 2):.5f})')
        print(f'AUC:{scores["test_auc"].mean():.5f} (+/- {(scores["test_auc"].std() * 2):.5f})')

    return


def data_block_n(i, batch_size):
    n_train = train_negative_frame.iloc[i * batch_size:i * batch_size + batch_size]
    block = pd.concat([n_train, train_positive_frame], axis=0)
    block_y = block.iloc[:, -1].values
    block_x = block.iloc[:, 0:-1].values
    # os_x, os_y = SMOTEN(random_state=42).fit_resample(block_x, block_y)
    # out_x, out_y = shuffle(os_x, os_y)
    return block_x, block_y


def main():
    os_rate = 0.6
    ir = float(len(train_negative_genes) / len(train_positive_genes))
    epoch_num = int(ir * os_rate)
    batch_size = int(len(train_negative_genes) / epoch_num)
    for i in tqdm.tqdm(range(epoch_num)):
        x, y = data_block_n(i, batch_size)
        train_one_epoch(x, y, i)
    mean_performance = {'Round': 'MEAN',
                        'Accuracy': f'{pd.to_numeric(cv_scorings["Accuracy"]).mean():.5f}',
                        'Precision': f'{pd.to_numeric(cv_scorings["Precision"]).mean():.5f}',
                        'Recall': f'{pd.to_numeric(cv_scorings["Recall"]).mean():.5f}',
                        'F1': f'{pd.to_numeric(cv_scorings["F1"]).mean():.5f}',
                        'AUC': f'{pd.to_numeric(cv_scorings["AUC"]).mean():.5f}'}
    cv_scorings.loc[len(cv_scorings)] = mean_performance
    cv_scorings.to_csv(os.path.join(CV_DIR, "AL_benchmark.csv"), index=False, sep="\t")

if __name__ == "__main__":
    main()


