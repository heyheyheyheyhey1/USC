import os
import pandas as pd
import numpy as np
import random
import pickle
from collections import namedtuple
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc

random.seed(716)

# Prepare test data
out_dir = "result"
t_dir = os.path.join(out_dir, 'test_data')
if not os.path.exists(t_dir):
    os.makedirs(t_dir)

# positive test set
positive_test = [line.rstrip('\n') for line in open(os.path.join(out_dir, "test_rnmts.txt"))]
# negative test set
negative_test = [line.rstrip('\n') for line in open(os.path.join(out_dir, "test_negative.txt"))]
# selected dataset
selected_dataset = pd.read_csv(os.path.join("data", "selected_dataset.tsv"), delimiter="\t", index_col=0,
                               low_memory=False)  # 在训练的时候选中的数据
test_dataset = selected_dataset.loc[selected_dataset.index.isin(positive_test + negative_test)]


# Define function for drawing negative examples chuncks from list of genes
def negative_sample_draw(gene_list, l=len(positive_test), n=0):
    """get the nth chunck of negative examples"""
    return (gene_list[n * l:n * l + l])


# Define function for creating test set based on draw n. Note: last column is the label
def test_set_n(n=0):
    negative_examples = negative_sample_draw(negative_test, l=len(positive_test), n=n)
    test_examples = positive_test + negative_examples  # 测试基因名

    test_dataset_n = test_dataset.loc[test_dataset.index.isin(test_examples)].copy()  # 测试数据集
    test_dataset_n['Targets'] = 0.0

    test_dataset_n2 = test_dataset_n.copy()
    test_dataset_n2.iloc[test_dataset_n.index.isin(test_examples)][-1] = 1.0

    for target in test_dataset_n.index.to_list():
        if target in positive_test:
            test_dataset_n.loc[target, 'Targets'] = 1.0
    random.seed(4)
    test_dataset_n = shuffle(test_dataset_n)

    test_data = test_dataset_n.iloc[:, 0:-1]  # 去掉最后一个标识位Targets

    # Export test dataset
    test_data.to_csv(os.path.join(t_dir, f'test_data_n{n}.csv'), sep=",", index=True, header=True)
    return (test_dataset_n)


# Define function for averaging prediction probabilities
def averaging_predictions(results):
    rnmt_pr = []
    for prediction in results['gene'].unique():
        rnmt_pr.append(
            {
                'Gene': prediction,
                'Avg_probability': results[results['gene'] == prediction][1].mean()
            }
        )
    return (pd.DataFrame(rnmt_pr).sort_values(by='Avg_probability', ascending=False))


# Define function to evaluate model m range r on test set n
def model_performance_test(m="GB", r=2, n=0):  # r代表训练出的第r个模型 n代表第n份测试数据

    # Create test data (X_test) and labels (y_test)
    test_set = test_set_n(n=n)  # 第n份数据 一共298份
    gene_names = np.array(test_set.index.to_list())
    X_test = test_set.iloc[:, 0:-1].values  # 样本
    y_test = test_set.iloc[:, -1].values  # 样本label

    n_predictions = {}

    cv_scorings = pd.DataFrame(columns=['Model_n', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])

    # Evaluate range r of models m 
    for i in range(r):
        # load model
        read_dir = os.path.join('tuning')
        model = pickle.load(open(os.path.join(read_dir, f'round_{i}.sav'), 'rb'))  # 读取GB模型第i次训练

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        pred = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, pred[:, 1])
        classifier_performance = {'Model_n': f'{m}_{i}',
                                  'Accuracy': f'{accuracy:.5f}',
                                  'Precision': f'{precision:.5f}',
                                  'Recall': f'{recall:.5f}',
                                  'F1': f'{f1:.5f}',
                                  'AUC': f'{roc_auc}'}
        cv_scorings.loc[len(cv_scorings)] = classifier_performance
    classifier_performance = {'Model_n': 'MEAN',
                              'Accuracy': f'{pd.to_numeric(cv_scorings["Accuracy"]).mean():.5f}',
                              'Precision': f'{pd.to_numeric(cv_scorings["Precision"]).mean():.5f}',
                              'Recall': f'{pd.to_numeric(cv_scorings["Recall"]).mean():.5f}',
                              'F1': f'{pd.to_numeric(cv_scorings["F1"]).mean():.5f}',
                              'AUC': f'{pd.to_numeric(cv_scorings["AUC"]).mean():.5f}'}
    cv_scorings.loc[len(cv_scorings)] = classifier_performance
    cv_scorings.to_csv(os.path.join(out_dir, f'{m}_test_performance.txt'), index=False, sep='\t')


for t in tqdm(range(299)):
    model_performance_test(m="SVM", r=174, n=t)
