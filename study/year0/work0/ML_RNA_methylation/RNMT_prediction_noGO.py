import os
import pandas as pd
import numpy as np
import random
import pickle
from collections import namedtuple
from random import sample
from sklearn.utils import shuffle
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, average_precision_score, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


random.seed(301)
def variance_threshold_selector(data, threshold=0.5):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]

out_dir = "RMT_results_noGO"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

cv_dir = os.path.join(out_dir, "CV")
if not os.path.exists(cv_dir):
    os.makedirs(cv_dir)
    
### Import data from data dir
# harmonizome dataset (.tsv format)
harmonizome_data = "harmonizome_data_combined_noGO.tsv"
dataset_matrix = os.path.join("data", harmonizome_data)
harmonizome_dataset = pd.read_csv(dataset_matrix, delimiter="\t", index_col=0, low_memory=False)

print("Initial data dimensions:", harmonizome_dataset.shape)


# List of RNA methyltransferases and partner proteins 
rnmts = "RNMT.list"
known_rnmts = [line.rstrip('\n') for line in open(os.path.join("data", rnmts))]


# Set aside 20% of known RNA methylation genes as test data
random.seed(0)
test_positive_examples = sample(known_rnmts, int(0.2 * len(known_rnmts)))

test_rnmts = open(os.path.join(out_dir, "test_rnmts.txt"), "w")
for rnmt in test_positive_examples:
    test_rnmts.write(rnmt)
    test_rnmts.write("\n")
test_rnmts.close()

positive_examples = list(set(known_rnmts).difference(set(test_positive_examples)))
#set(positive_examples + test_positive_examples) == set(known_rnmts)

training_rnmts = open(os.path.join(out_dir, "training_rnmts.txt"), "w")
for rnmt in positive_examples:
    training_rnmts.write(rnmt)
    training_rnmts.write("\n")
training_rnmts.close()

# All human genes
all_genes = harmonizome_dataset.index.values.tolist()
harmonizome_genes = open(os.path.join(out_dir, "all_genes.txt"), "w")
for gene in all_genes:
    harmonizome_genes.write(gene)
    harmonizome_genes.write("\n")
harmonizome_genes.close()

# Set aside 20% of human genes as test data (negative class)
random.seed(1)
other_genes = list(set(all_genes).difference(set(known_rnmts)))
other_genes = shuffle(other_genes)

random.seed(2)
test_negative_examples = sample(other_genes, int(0.2 * len(other_genes)))
test_neg = open(os.path.join(out_dir, "test_negative.txt"), "w")
for negative in test_negative_examples:
    test_neg.write(negative)
    test_neg.write("\n")
test_neg.close()

negative_genes = list(set(other_genes).difference(set(test_negative_examples)))
# print(set(negative_genes + test_negative_examples) == set(other_genes))
training_neg = open(os.path.join(out_dir, "training_negative.txt"), "w")
for negative in negative_genes:
    training_neg.write(negative)
    training_neg.write("\n")
training_neg.close()

positive_examples_in_all_genes = list(set(all_genes).intersection(set(positive_examples)))
num_positive_examples = len(positive_examples_in_all_genes)

print("Number of known RNMT:", len(known_rnmts))
print("Number of positive examples for training/cv:", num_positive_examples)
print("Number of positive examples for testing:", len(test_positive_examples))
print("Number of other remaining genes:", len(other_genes))
print("Number of other genes for training/cv:", len(negative_genes))
print("Number of other genes for testing:", len(test_negative_examples))
print("Number of draws of negative examples for balanced sampling:", len(negative_genes) / num_positive_examples)


# Define function for drawing negative examples chuncks from list of genes
def negative_sample_draw(gene_list, l = num_positive_examples, n=0):
    """get the nth chunck of negative examples"""
    return(gene_list[n*l:n*l+l])

### Prefiltering - Feature selection
# Feature selection based on prefiltering across all chunks of negative examples
def feature_selection(c=289):
    for n in range(c):
        negative_examples = negative_sample_draw(negative_genes, n=n)
        training_examples = positive_examples_in_all_genes + negative_examples
        dataset = harmonizome_dataset.loc[harmonizome_dataset.index.isin(training_examples)]
        dataset = dataset.dropna(axis=1, thresh=(len(dataset)*0.3))
        dataset = dataset.fillna(0)
        # remove all features that are either one or zero (on or off) in more than 80% of the samples
        dataset = variance_threshold_selector(dataset, threshold=.8 * (1 - .8))
        for feature in dataset.columns.to_list():
            yield(feature)

features_to_keep = list({*feature_selection()})

feature_file = open(os.path.join(out_dir, 'selected_features.txt'), "w")
for feature in features_to_keep:
    feature_file.write(feature)
    feature_file.write("\n")
feature_file.close()

selected_dataset = harmonizome_dataset[features_to_keep]
selected_dataset = selected_dataset.fillna(0)
selected_dataset.to_csv(os.path.join(out_dir, 'selected_dataset.tsv'), sep="\t", index=True, header=True)

### Functions for hyper-parameter tuning
def SVM_tuning(n, n_dir, X_train, y_train):
    scores = ['accuracy'] # select scores e.g scores = ['recall', 'accuracy']
    grid_param_svm = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    svm_tuning_info = open(os.path.join(n_dir, f'SVM_tuning_n{n}.txt'), "w")
    for score in scores:
        svm_tuning = GridSearchCV(SVC(random_state=3), grid_param_svm, cv=KFold(3, shuffle=True, random_state=3), scoring='%s' % score, n_jobs=8)
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
    return(svm_tuning.best_params_)

def RF_tuning(n, n_dir, X_train, y_train):
    scores = ['accuracy'] # select scores e.g scores = ['recall', 'accuracy']
    n_estimators = [500, 1000, 1500, 2500, 5000]
    grid_param_rf = {'n_estimators': n_estimators}

    rf_tuning_info = open(os.path.join(n_dir, f'RF_tuning_n{n}.txt'), "w")
    for score in scores:
        rf_tuning = GridSearchCV(RandomForestClassifier(random_state=3), grid_param_rf, cv=KFold(3, shuffle=True, random_state=3), scoring='%s' % score, n_jobs=8)
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

    max_features = ['auto', 'sqrt', 'log2']
    max_depth = [10, 20, 30, 40, 50]
    max_depth.append(None)
    min_samples_split = [2, 5, 10, 15, 20]
    min_samples_leaf = [1, 2, 5, 10, 15]
    grid_param_rf_random = {'max_features': max_features,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf}
    for score in scores:
        rf_random = RandomizedSearchCV(estimator=RandomForestClassifier(n_estimators = rf_tuning.best_params_['n_estimators'], random_state=3), param_distributions=grid_param_rf_random, n_iter=100, cv=KFold(3, shuffle=True, random_state=3), verbose=0, scoring='%s' % score, n_jobs=8)
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
    return(rf_tuning.best_params_)

def GM_tuning(n, n_dir, X_train, y_train):
    scores = ['accuracy'] # select scores e.g scores = ['recall', 'accuracy']
    n_estimators = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    learning_rate = [0.05, 0.1, 0.2]
    grid_param_gbm = {'n_estimators': n_estimators,
                     'learning_rate': learning_rate}

    gbm_tuning_info = open(os.path.join(n_dir, f'GB_tuning_n{n}.txt'), "w")
    for score in scores:
        gbm_tuning = GridSearchCV(GradientBoostingClassifier(subsample=0.8, random_state=3), grid_param_gbm, cv=KFold(3, shuffle=True, random_state=3), scoring='%s' % score, n_jobs=8)
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

    max_features = ['auto', 'sqrt', 'log2']
    max_depth = [3, 5, 8, 10, 20]
    min_samples_split = [2, 5, 10, 15, 20]
    min_samples_leaf = [1, 2, 5, 10, 15]
    grid_param_gmb_random = {'max_features': max_features,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf}

    for score in scores:
        gbm_random = RandomizedSearchCV(estimator=GradientBoostingClassifier(n_estimators = gbm_tuning.best_params_['n_estimators'], subsample=0.8, random_state=3), param_distributions=grid_param_gmb_random, n_iter=100, cv=KFold(3, shuffle=True, random_state=3), verbose=0, scoring='%s' % score, n_jobs=8)
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
    return(gbm_tuning.best_params_)


### Model training for each training example set
def model_training_n(n=0):
    n_dir = os.path.join(out_dir, f'draw_{n}')
    if not os.path.exists(n_dir):
        os.makedirs(n_dir)

    negative_examples = negative_sample_draw(negative_genes, n=n)
    training_examples = positive_examples_in_all_genes + negative_examples

    # Export negative class samples in chunck
    negative_example_file = open(os.path.join(n_dir, f'negative_class_n{n}.txt'), "w")
    for negative_example in negative_examples:
        negative_example_file.write(negative_example)
        negative_example_file.write("\n")
    negative_example_file.close()

    # Prepare training dataset
    train_val_dataset = selected_dataset.loc[selected_dataset.index.isin(training_examples)].copy()
    train_val_dataset['Targets'] = 0.0
    for target in train_val_dataset.index.to_list():
        if target in positive_examples_in_all_genes:
            train_val_dataset.loc[target, 'Targets'] = 1.0
    random.seed(4)
    train_val_dataset = shuffle(train_val_dataset)

    # Double-check that the training dataset does not contain labels
    train_val_data = train_val_dataset.iloc[:, 0:-1]
    for i in range(len(train_val_data.columns)):
        data = abs(train_val_data.iloc[:, i])
        if data.equals(train_val_dataset.iloc[:, -1]):
            raise Exception("Chunk n:", c, "target labels match feature:", i, train_val_data.columns[i], "nFeatures: ", train_val_data.shape[1])

    # Export training dataset
    train_val_data.to_csv(os.path.join(n_dir, f'training_data_n{n}.csv'), sep=",", index=True, header=True)

    # Create training data (X_train) and labels (y_train)
    X_train = train_val_dataset.iloc[:, 0:-1].values
    y_train = train_val_dataset.iloc[:, -1].values
    
    ### Hyper-parameter tuning   
    svm_tuning = SVM_tuning(n=n, n_dir=n_dir, X_train=X_train, y_train=y_train)
    rf_tuning = RF_tuning(n=n, n_dir=n_dir, X_train=X_train, y_train=y_train)
    gm_tuning = GM_tuning(n=n, n_dir=n_dir, X_train=X_train, y_train=y_train)

    ### Define and train classifiers
    LR_model = LogisticRegression(solver='lbfgs', random_state=5)
    GNB_model = GaussianNB()
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

    # Model training and performance estimates based on 10-fold cross-validation
    scorings = {'accuracy' : make_scorer(accuracy_score), 
                'recall' : make_scorer(recall_score), 
                'precision' : make_scorer(precision_score),
                'f1_score' : make_scorer(f1_score),
                'auc': 'roc_auc'}

    cv_scorings = pd.DataFrame(columns=('Classifier', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'))

    ## Train all models and evaluate with cross-validation
    for classifier, name in [(LR_model, 'LR'),
                             (GNB_model, 'GNB'),
                             (SVM_model, 'SVM'),
                             (RFC_model, 'RF'),
                             (GBM_model, 'GB')]:
        scores = cross_validate(estimator=classifier, X=X_train, y=y_train, cv=10, scoring=scorings, n_jobs=8)
        classifier_performance = {'Classifier': name,
                                  'Accuracy': f'{scores["test_accuracy"].mean():.5f} (+/- {(scores["test_accuracy"].std() * 2):.5f})',
                                  'Precision': f'{scores["test_precision"].mean():.5f} (+/- {(scores["test_precision"].std() * 2):.5f})', 
                                  'Recall': f'{scores["test_recall"].mean():.5f} (+/- {(scores["test_recall"].std() * 2):.5f})', 
                                  'F1': f'{scores["test_f1_score"].mean():.5f} (+/- {(scores["test_f1_score"].std() * 2):.5f})',
                                  'AUC': f'{scores["test_auc"].mean():.5f} (+/- {(scores["test_auc"].std() * 2):.5f})'}
        cv_scorings = cv_scorings.append(pd.Series(classifier_performance), ignore_index=True)

        # Fit and save model
        model = classifier.fit(X_train, y_train)
        pickle.dump(model, open(os.path.join(n_dir, f'{name}_m{n}.sav'), 'wb'))

        # Export feature ranking
        if name in ['RF', 'GB']:
            feature_ranking = pd.Series(model.feature_importances_, index=features_to_keep).sort_values(ascending=False)
            sorted_features = feature_ranking
            sorted_features.to_csv(os.path.join(n_dir, f'{name}_f_ranking_n{n}.tsv'), index=True, sep=",", header=False)

        # Export model predictions on training set
        predicted_train = classifier.predict_proba(X_train)
        np.savetxt(os.path.join(n_dir, f'{name}_pr{n}_train.csv'), predicted_train, delimiter=",", header="0,1")
        
        # Export model predictions of all human genes
        random.seed(6)
        full_dataset = shuffle(selected_dataset)
        predicted_all = classifier.predict_proba(full_dataset)
        predictions = pd.DataFrame(predicted_all, columns=['0', '1'], index=full_dataset.index.to_list())
        predictions.to_csv(os.path.join(n_dir, f'{name}_pr{n}_all_genes.csv'), index=True, sep=",", header=True)

    # Export cross-validation performance table    
    cv_scorings.to_csv(os.path.join(cv_dir, f'cv_scorings_n{n}.tsv'), index=False, sep="\t")

for i in range(289):
    model_training_n(n=i)

