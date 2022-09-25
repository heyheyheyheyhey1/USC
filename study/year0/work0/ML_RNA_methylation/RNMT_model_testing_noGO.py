import os
import pandas as pd
import numpy as np
import random
import pickle
from collections import namedtuple
from random import sample
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, average_precision_score, auc
from sklearn.metrics import roc_auc_score
from statistics import mean, stdev


random.seed(716)

# Prepare test data
out_dir = "RMT_results_noGO"
t_dir = os.path.join(out_dir, 'test_data')
if not os.path.exists(t_dir):
    os.makedirs(t_dir)

# positive test set
positive_test = [line.rstrip('\n') for line in open(os.path.join(out_dir, "test_rnmts.txt"))]
# negative test set
negative_test = [line.rstrip('\n') for line in open(os.path.join(out_dir, "test_negative.txt"))]
# selected dataset
selected_dataset = pd.read_csv(os.path.join(out_dir, "selected_dataset.tsv"), delimiter="\t", index_col=0, low_memory=False)
test_dataset = selected_dataset.loc[selected_dataset.index.isin(positive_test + negative_test)]


# Define function for drawing negative examples chuncks from list of genes
def negative_sample_draw(gene_list, l = len(positive_test), n=0):
    """get the nth chunck of negative examples"""
    return(gene_list[n*l:n*l+l])


# Define function for creating test set based on draw n. Note: last column is the label
def test_set_n(n=0):
    negative_examples = negative_sample_draw(negative_test, l=len(positive_test), n=n)
    test_examples = positive_test + negative_examples

    test_dataset_n = test_dataset.loc[test_dataset.index.isin(test_examples)].copy()
    test_dataset_n['Targets'] = 0.0
    for target in test_dataset_n.index.to_list():
        if target in positive_test:
            test_dataset_n.loc[target, 'Targets'] = 1.0
    random.seed(4)
    test_dataset_n = shuffle(test_dataset_n)

    # Double-check that the test dataset does not contain labels
    test_data = test_dataset_n.iloc[:, 0:-1]
    for i in range(len(test_data.columns)):
        data = abs(test_data.iloc[:, i])
        if data.equals(test_dataset_n.iloc[:, -1]):
            raise Exception("Chunk n:", n, "target labels match feature:", i, test_data.columns[i], "nFeatures: ", test_data.shape[1])

    # Export test dataset
    test_data.to_csv(os.path.join(t_dir, f'test_data_n{n}.csv'), sep=",", index=True, header=True)
    return(test_dataset_n)


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
    return(pd.DataFrame(rnmt_pr).sort_values(by='Avg_probability', ascending=False))
    

# Define function to evaluate model m range r on test set n
def model_performance_test(m = "GB", r = 2, n = 0):
    
    # Create test data (X_test) and labels (y_test)
    test_set = test_set_n(n=n)
    gene_names = np.array(test_set.index.to_list())
    X_test = test_set.iloc[:, 0:-1].values
    y_test = test_set.iloc[:, -1].values

    
    n_predictions = {}
    model_run = namedtuple("model_run", ["model", "sample"])

    # Evaluate range r of models m 
    for i in range(r):
        # load model
        read_dir = os.path.join(out_dir, f'draw_{i}')
        model = pickle.load(open(os.path.join(read_dir, f'{m}_m{i}.sav'), 'rb'))
        
        # predictions and perfomance metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision= precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        pred = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, pred[:,1])
        
        # save to pd dataframe
        test_predictions = pd.DataFrame(pred)
        test_predictions['gene'] = gene_names
        test_predictions['model'] = m
        test_predictions['sample'] = i
        test_predictions = test_predictions[['gene', 0, 1, 'model', 'sample']]
        test_predictions['Accuracy'] = '{0:0.5f}'.format(accuracy)
        test_predictions['Precision'] = '{0:0.5f}'.format(precision)
        test_predictions['Recall'] = '{0:0.5f}'.format(recall)
        test_predictions['F1'] = '{0:0.5f}'.format(f1)
        test_predictions['ROC_AUC'] = '{0:0.5f}'.format(roc_auc)

        n_predictions[model_run(m, i)] = test_predictions

    # export results tables    
    df_predictions = pd.concat(n_predictions.values(), sort=False, join='outer', axis=0, ignore_index=True)
    df_predictions.to_csv(os.path.join(t_dir, f'{m}_test_n{n}_full.tsv'), index=False, sep="\t")

    predictions = averaging_predictions(df_predictions)
    predictions.to_csv(os.path.join(t_dir, f'{m}_test_n{n}_avg.tsv'), index=False, sep="\t")


for t in range(297):
    model_performance_test(m = "SVM", r = 289, n = t)


