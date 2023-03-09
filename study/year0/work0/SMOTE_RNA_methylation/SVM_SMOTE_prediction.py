import random

import pandas as pd
import pickle
import os
from random import sample
from sklearn.utils import shuffle
import tqdm
from imblearn.over_sampling import *
from sklearn.model_selection import *
from sklearn.svm import SVC
from sklearn.metrics import *

harmonizome_data = "harmonizome_data_combined.tsv"
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

TUNING_DIR = os.path.join("tuning")
RESULT_DIR = os.path.join("result")
if not os.path.exists(TUNING_DIR):
    os.makedirs(TUNING_DIR)

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)


def SVM_tuning(n, X_train, y_train):
    scores = ['accuracy']  # select scores e.g scores = ['recall', 'accuracy']
    grid_param_svm = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4], 'C': [1, 10, 50, 100, 500, 1000,1500,2000,2500]},
                      {'kernel': ['linear'], 'C': [1, 10, 50, 100, 500, 1000,1500,2000,2500]}]
    svm_tuning_info = open(os.path.join("tuning", f'SVM_tuning_{n}.txt'), "w")
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
    # pred_y = SVM_model.predict(all_x)
    # accuracy = accuracy_score(all_y,pred_y)
    # precision = precision_score(all_y, pred_y)
    # recall = recall_score(all_y, pred_y)
    # f1 = f1_score(all_y, pred_y)

    pickle.dump(model, open(os.path.join(TUNING_DIR, f'round_{n}.sav'), 'wb'))
    if (n % 3 == 0):
        # print(f"round_{n + 1} result :\n")
        # print(f'accuracy:{accuracy}')
        # print(f'precision:{precision}')
        # print(f'recall:{recall}')
        # print(f'f1:{f1}')
        print(f'accuracy:{scores["test_accuracy"].mean():.5f} (+/- {(scores["test_accuracy"].std() * 2):.5f})')
        print(f'Precision:{scores["test_precision"].mean():.5f} (+/- {(scores["test_precision"].std() * 2):.5f})')
        print(f'Recall:{scores["test_recall"].mean():.5f} (+/- {(scores["test_recall"].std() * 2):.5f})')
        print(f'F1:{scores["test_f1_score"].mean():.5f} (+/- {(scores["test_f1_score"].std() * 2):.5f})')
        print(f'AUC:{scores["test_auc"].mean():.5f} (+/- {(scores["test_auc"].std() * 2):.5f})')

    return model


def data_block_n(i, batch_size):
    n_train = train_negative_frame.iloc[i * batch_size:i * batch_size + batch_size]
    block = pd.concat([n_train,train_positive_frame],axis=0)
    block_y = block.iloc[:, -1].values
    block_x = block.iloc[:, 0:-1].values
    os_x,os_y = SMOTEN(random_state=42).fit_resample(block_x, block_y)
    out_x, out_y = shuffle(os_x,os_y)
    return out_x, out_y


def main():
    os_rate = 0.6
    ir = float(len(train_negative_genes) / len(train_positive_genes))
    epoch_num = int(ir * os_rate)
    batch_size = int(len(train_negative_genes) / epoch_num)
    for i in tqdm.tqdm(range(epoch_num)):
        x, y = data_block_n(i, batch_size)
        train_one_epoch(x, y, i)


if __name__ == "__main__":
    main()
