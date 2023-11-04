import random
import torch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

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

CV_DIR = os.path.join("CV")
MODEL_DIR = os.path.join("model")
TUNING_DIR = os.path.join("TUNNING")
DATA_DIR = os.path.join("data")
LATENT_DIM = 128
CLASSIFIER_NAMES = ['SVM', 'GB', 'GNB', 'LR', 'RF']
# CLASSIFIER_NAMES = ['SVM']
generator = Generator(in_dim=LATENT_DIM, out_dim=1517)
generator.load_state_dict(torch.load(os.path.join(MODEL_DIR, "wgangp", "generator", "generator_n_574_acc_0.494.pth")))
generator.eval()

if not os.path.exists(CV_DIR):
    os.makedirs(CV_DIR)

for name in CLASSIFIER_NAMES:
    if not os.path.exists(os.path.join(MODEL_DIR, name)):
        os.makedirs(os.path.join(MODEL_DIR, name))
    if not os.path.exists(os.path.join(TUNING_DIR, name)):
        os.makedirs(os.path.join(TUNING_DIR, name))

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

cv_scorings = pd.DataFrame(columns=['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])


def SVM_tuning(n, X_train, y_train):
    scores = ['accuracy']  # select scores e.g scores = ['recall', 'accuracy']
    # grid_param_svm = [
    #     {'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4], 'C': np.arange(0, 1501, 100)[1:]},
    #     {'kernel': ['linear'], 'C': np.arange(0, 1501, 100)[1:]}, ]
    grid_param_svm = [
        {'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
        {'kernel': ['linear'],'C': [1, 10, 100, 1000]}]
    svm_tuning_info = open(os.path.join(TUNING_DIR,'SVM', f'SVM_tuning_{n}.txt'), "w")
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


def RF_tuning(n, X_train, y_train):
    scores = ['accuracy']  # select scores e.g scores = ['recall', 'accuracy']
    n_estimators = [500, 1000, 1500, 2500, 5000]
    grid_param_rf = {'n_estimators': n_estimators}
    if not os.path.exists(os.path.join(TUNING_DIR, "RF")):
        os.makedirs(os.path.join(TUNING_DIR, "RF"))
    rf_tuning_info = open(os.path.join(TUNING_DIR, "RF", f'RF_tuning_n{n}.txt'), "w")
    for score in scores:
        rf_tuning = GridSearchCV(RandomForestClassifier(random_state=3), grid_param_rf,
                                 cv=KFold(3, shuffle=True, random_state=3), scoring='%s' % score, n_jobs=12)
        rf_tuning.fit(X_train, y_train)

        print("# Tuning hyper-parameters for %s" % score, file=rf_tuning_info)
        print("Best parameters set found on training set:", file=rf_tuning_info)
        print(rf_tuning.best_params_, file=rf_tuning_info)
        print(file=rf_tuning_info)
        print("Grid scores on training set:", file=rf_tuning_info)
        means = rf_tuning.cv_results_['mean_test_score']
        stds = rf_tuning.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, rf_tuning.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params), file=rf_tuning_info)
        print(file=rf_tuning_info)

    print("Selected parameters:", file=rf_tuning_info)
    print(rf_tuning.best_params_, file=rf_tuning_info)
    print(file=rf_tuning_info)

    max_features = ['sqrt', 'log2']
    max_depth = [10, 20, 30, 40, 50]
    max_depth.append(None)
    min_samples_split = [2, 5, 10, 15, 20]
    min_samples_leaf = [1, 2, 5, 10, 15]
    grid_param_rf_random = {'max_features': max_features,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf}
    for score in scores:
        rf_random = RandomizedSearchCV(
            estimator=RandomForestClassifier(n_estimators=rf_tuning.best_params_['n_estimators'], random_state=3),
            param_distributions=grid_param_rf_random, n_iter=100, cv=KFold(3, shuffle=True, random_state=3), verbose=0,
            scoring='%s' % score, n_jobs=12)
        rf_random.fit(X_train, y_train)

        print("# Randomized search for other hyper-parameters", file=rf_tuning_info)
        print("Best parameters set found on training set:", file=rf_tuning_info)
        print(rf_random.best_params_, file=rf_tuning_info)
        print(file=rf_tuning_info)
        print("Randomized search scores on training set:", file=rf_tuning_info)
        means = rf_random.cv_results_['mean_test_score']
        stds = rf_random.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, rf_random.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params), file=rf_tuning_info)
        print(file=rf_tuning_info)

    print("Selected parameters:", file=rf_tuning_info)
    print(rf_random.best_params_, file=rf_tuning_info)
    rf_tuning_info.close()

    rf_tuning.best_params_.update(rf_random.best_params_)
    return (rf_tuning.best_params_)


def GM_tuning(n, X_train, y_train):
    scores = ['accuracy']  # select scores e.g scores = ['recall', 'accuracy']
    n_estimators = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    learning_rate = [0.05, 0.1, 0.2]
    grid_param_gbm = {'n_estimators': n_estimators,
                      'learning_rate': learning_rate}
    if not os.path.exists(os.path.join(TUNING_DIR, "GB")):
        os.makedirs(os.path.join(TUNING_DIR, "GB"))
    gbm_tuning_info = open(os.path.join(TUNING_DIR, "GB", f'GB_tuning_n{n}.txt'), "w")
    for score in scores:
        gbm_tuning = GridSearchCV(GradientBoostingClassifier(subsample=0.8, random_state=3), grid_param_gbm,
                                  cv=KFold(3, shuffle=True, random_state=3), scoring='%s' % score, n_jobs=12)
        gbm_tuning.fit(X_train, y_train)

        print("# Tuning hyper-parameters for %s" % score, file=gbm_tuning_info)
        print("Best parameters set found on training set:", file=gbm_tuning_info)
        print(gbm_tuning.best_params_, file=gbm_tuning_info)
        print(file=gbm_tuning_info)
        print("Grid scores on training set:", file=gbm_tuning_info)
        means = gbm_tuning.cv_results_['mean_test_score']
        stds = gbm_tuning.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, gbm_tuning.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params), file=gbm_tuning_info)
        print(file=gbm_tuning_info)

    print("Selected parameters:", file=gbm_tuning_info)
    print(gbm_tuning.best_params_, file=gbm_tuning_info)
    print(file=gbm_tuning_info)

    max_features = ['sqrt', 'log2']
    max_depth = [3, 5, 8, 10, 20]
    min_samples_split = [2, 5, 10, 15, 20]
    min_samples_leaf = [1, 2, 5, 10, 15]
    grid_param_gmb_random = {'max_features': max_features,
                             'max_depth': max_depth,
                             'min_samples_split': min_samples_split,
                             'min_samples_leaf': min_samples_leaf}

    for score in scores:
        gbm_random = RandomizedSearchCV(
            estimator=GradientBoostingClassifier(n_estimators=gbm_tuning.best_params_['n_estimators'], subsample=0.8,
                                                 random_state=3), param_distributions=grid_param_gmb_random, n_iter=100,
            cv=KFold(3, shuffle=True, random_state=3), verbose=0, scoring='%s' % score, n_jobs=12)
        gbm_random.fit(X_train, y_train)

        print("# Randomized search for other hyper-parameters", file=gbm_tuning_info)
        print("Best parameters set found on training set:", file=gbm_tuning_info)
        print(gbm_random.best_params_, file=gbm_tuning_info)
        print(file=gbm_tuning_info)
        print("Randomized search scores on training set:", file=gbm_tuning_info)
        means = gbm_random.cv_results_['mean_test_score']
        stds = gbm_random.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, gbm_random.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params), file=gbm_tuning_info)
        print(file=gbm_tuning_info)

    print("Selected parameters:", file=gbm_tuning_info)
    print(gbm_random.best_params_, file=gbm_tuning_info)
    gbm_tuning_info.close()

    gbm_tuning.best_params_.update(gbm_random.best_params_)
    return (gbm_tuning.best_params_)


def train_one_epoch(X, Y, n):
    svm_tuning = SVM_tuning(n, X, Y)
    rf_tuning = RF_tuning(n, X, Y)
    gm_tuning = GM_tuning(n, X, Y)
    LR_model = LogisticRegression(solver='lbfgs', random_state=5)
    GNB_model = GaussianNB()
    scorings = {'accuracy': make_scorer(accuracy_score),
                'recall': make_scorer(recall_score),
                'precision': make_scorer(precision_score),
                'f1_score': make_scorer(f1_score),
                'auc': 'roc_auc'}
    if svm_tuning['kernel'] == 'rbf':
        SVM_model = SVC(C=svm_tuning['C'],
                        gamma=svm_tuning['gamma'],
                        kernel=svm_tuning['kernel'],
                        probability=True,
                        random_state=5)
    elif svm_tuning['kernel'] == 'linear':
        SVM_model = SVC(C=svm_tuning['C'],
                        kernel=svm_tuning['kernel'],
                        probability=True,
                        random_state=5)

    RFC_model = RandomForestClassifier(n_estimators=rf_tuning['n_estimators'],
                                       min_samples_split=rf_tuning['min_samples_split'],
                                       min_samples_leaf=rf_tuning['min_samples_leaf'],
                                       max_features=rf_tuning['max_features'],
                                       max_depth=rf_tuning['max_depth'],
                                       random_state=5)
    GBM_model = GradientBoostingClassifier(n_estimators=gm_tuning['n_estimators'],
                                           learning_rate=gm_tuning['learning_rate'],
                                           subsample=0.8,
                                           min_samples_split=gm_tuning['min_samples_split'],
                                           min_samples_leaf=gm_tuning['min_samples_leaf'],
                                           max_features=gm_tuning['max_features'],
                                           max_depth=gm_tuning['max_depth'],
                                           random_state=5)

    for classifier, name in [(LR_model, 'LR'),(GNB_model, 'GNB'),(SVM_model, 'SVM'),(RFC_model, 'RF'),(GBM_model, 'GB')]:
    # for classifier, name in [(SVM_model, 'SVM')]:
        scores = cross_validate(estimator=classifier, X=X, y=Y, cv=10, scoring=scorings, n_jobs=12)
        model = classifier.fit(X, Y)

        pickle.dump(model, open(os.path.join(MODEL_DIR,name, f'round_{n}.sav'), 'wb'))
        classifier_performance = {'Classifier': name,
                                  'Accuracy': float(f'{scores["test_accuracy"].mean():.5f}'),
                                  'Precision': float(f'{scores["test_precision"].mean():.5f}'),
                                  'Recall': float(f'{scores["test_recall"].mean():.5f}'),
                                  'F1': float(f'{scores["test_f1_score"].mean():.5f} '),
                                  'AUC': float(f'{scores["test_auc"].mean():.5f}')}
        cv_scorings.loc[len(cv_scorings)] = classifier_performance
    return


def oversample(x, y):
    synthetic_num = abs(np.sum(y == 0) - np.sum(y == 1))
    with torch.no_grad():
        synthetic_data = generator(torch.rand(synthetic_num, LATENT_DIM))
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

    for name in CLASSIFIER_NAMES:
        cv_pd = cv_scorings[cv_scorings["Classifier"] == name].copy()
        cv_pd.loc[len(cv_pd)] = cv_pd.mean(numeric_only=True)
        cv_pd.to_csv(os.path.join(CV_DIR, f"{name}_GAN_CV.csv"), index=False, sep="\t")


if __name__ == "__main__":
    main()
