import random
import torch
from WGANGP import Generator
import pandas as pd
import pickle
import os
from sklearn.utils import shuffle
import tqdm
import numpy as np
from sklearn.model_selection import *
from sklearn.svm import SVC
from sklearn.metrics import *

CV_DIR = os.path.join("CV", "SVM")
MODEL_DIR = os.path.join("model")
TUNING_DIR = os.path.join("CV", "SVM", "tuning")
DATA_DIR = os.path.join("data")
generator = Generator(in_dim=128, out_dim=1517)
generator.load_state_dict(torch.load(os.path.join(MODEL_DIR, "wgangp", "generator", "generator_n_1355_acc_0.459.pth")))
generator.eval()

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

if not os.path.exists(CV_DIR):
    os.makedirs(CV_DIR)

if not os.path.exists(TUNING_DIR):
    os.makedirs(TUNING_DIR)

harmonizome_data = "harmonizome_data_combined.tsv"
selected_data = "selected_dataset.tsv"
rnmts = "RNMT.list"
dataset_matrix = os.path.join("data", selected_data)
selected_data = pd.read_csv(dataset_matrix, delimiter="\t", index_col=0, low_memory=False)
all_genes = selected_data.index.unique().to_list()
positive_genes = [line.rstrip('\n') for line in open(os.path.join("data", rnmts))]
negative_genes = set(all_genes).difference(positive_genes)  # 负样本

selected_data["Y"] = [1 if idx in positive_genes else 0 for idx in selected_data.index.to_list()]
test_positive_genes = [line.rstrip('\n') for line in open(os.path.join("data", "test_positive_genes.txt"))]
test_negative_genes = [line.rstrip('\n') for line in open(os.path.join("data", "test_negative_genes.txt"))]
train_positive_genes = set(positive_genes).difference(test_positive_genes)
train_negative_genes = set(negative_genes).difference(test_negative_genes)
train_positive_frame = selected_data.loc[list(train_positive_genes)]
train_negative_frame = selected_data.loc[list(train_negative_genes)]

print("all genes num: %d\n" % (len(all_genes)))
print("test genes num: %d\n" % (len(test_negative_genes) + len(test_positive_genes)))

random.seed(42)

cv_scorings = pd.DataFrame(columns=['Round', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])


def SVM_tuning(n, X_train, y_train):
    scores = ['accuracy']  # select scores e.g scores = ['recall', 'accuracy']
    grid_param_svm = [
        {'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4], 'C': np.arange(0, 1501, 100)[1:],
         "class_weight": ["balanced"]},
        {'kernel': ['linear'], 'C': np.arange(0, 1501, 100)[1:], "class_weight": ["balanced"]}, ]
    svm_tuning_info = open(os.path.join(TUNING_DIR, f'SVM_tuning_{n}.txt'), "w")
    for score in scores:
        svm_tuning = GridSearchCV(SVC(random_state=3), grid_param_svm, cv=KFold(3, shuffle=True, random_state=3),
                                  scoring='%s' % score, n_jobs=12)
        svm_tuning.fit(X_train, y_train)

        print("# Tuning hyper-parameters for %s" % score, file=svm_tuning_info)
        print("Best parameters set found on training set:", file=svm_tuning_info)
        print(svm_tuning.best_params_, file=svm_tuning_info)
        print(file=svm_tuning_info)
        print("Grid scores on training set:", file=svm_tuning_info)
        means = svm_tuning.cv_results_['mean_test_score']
        stds = svm_tuning.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, svm_tuning.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params), file=svm_tuning_info)
        print(file=svm_tuning_info)

    print("Selected parameters:", file=svm_tuning_info)
    print(svm_tuning.best_params_, file=svm_tuning_info)
    svm_tuning_info.close()
    return (svm_tuning.best_params_)


def train_one_epoch(X, Y, n):
    tuning = SVM_tuning(n, X, Y)
    scorings = {'accuracy': make_scorer(accuracy_score),
                'recall': make_scorer(recall_score),
                'precision': make_scorer(precision_score),
                'f1_score': make_scorer(f1_score),
                'auc': 'roc_auc'}
    if tuning['kernel'] == 'rbf':
        SVM_model = SVC(C=tuning['C'],
                        gamma=tuning['gamma'],
                        kernel=tuning['kernel'],
                        probability=True,
                        random_state=5)
    elif tuning['kernel'] == 'linear':
        SVM_model = SVC(C=tuning['C'],
                        kernel=tuning['kernel'],
                        probability=True,
                        random_state=5)
    scores = cross_validate(estimator=SVM_model, X=X, y=Y, cv=10, scoring=scorings, n_jobs=12)
    model = SVM_model.fit(X, Y)

    pickle.dump(model, open(os.path.join(MODEL_DIR, "SVM", f'round_{n}.sav'), 'wb'))
    classifier_performance = {'Round': n,
                              'Accuracy': f'{scores["test_accuracy"].mean():.5f}',
                              'Precision': f'{scores["test_precision"].mean():.5f}',
                              'Recall': f'{scores["test_recall"].mean():.5f}',
                              'F1': f'{scores["test_f1_score"].mean():.5f} ',
                              'AUC': f'{scores["test_auc"].mean():.5f}'}
    cv_scorings.loc[len(cv_scorings)] = classifier_performance

    if (n % 3 == 0):
        # print(f"round_{n + 1} result :\n")
        # print(f'accuracy:{accuracy}')
        # print(f'precision:{precision}')
        # print(f'recall:{recall}')
        # print(f'f1:{f1}')
        print(f'\naccuracy:{scores["test_accuracy"].mean():.5f} (+/- {(scores["test_accuracy"].std() * 2):.5f})')
        print(f'Precision:{scores["test_precision"].mean():.5f} (+/- {(scores["test_precision"].std() * 2):.5f})')
        print(f'Recall:{scores["test_recall"].mean():.5f} (+/- {(scores["test_recall"].std() * 2):.5f})')
        print(f'F1:{scores["test_f1_score"].mean():.5f} (+/- {(scores["test_f1_score"].std() * 2):.5f})')
        print(f'AUC:{scores["test_auc"].mean():.5f} (+/- {(scores["test_auc"].std() * 2):.5f})')

    return model


def oversample(x, y):
    synthetic_num = abs(np.sum(y == 0) - np.sum(y == 1))
    synthetic_data = generator(torch.rand(synthetic_num, 128))
    x_os = np.concatenate([synthetic_data.detach(), x]).copy()
    y_os = np.concatenate([np.ones([synthetic_num, ]), y]).copy()
    return x_os, y_os


def data_block_n(i, batch_size):
    n_train = train_negative_frame.iloc[i * batch_size:i * batch_size + batch_size]
    block = pd.concat([n_train, train_positive_frame], axis=0)
    block_y = block.iloc[:, -1].values
    block_x = block.iloc[:, 0:-1].values
    os_x, os_y = oversample(block_x, block_y)
    out_x, out_y = shuffle(os_x, os_y)
    return out_x, out_y


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
    cv_scorings.to_csv(os.path.join(CV_DIR, "SVM_benchmark.csv"), index=False, sep="\t")


if __name__ == "__main__":
    main()
