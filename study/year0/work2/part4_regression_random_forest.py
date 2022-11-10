import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/sars-3c-like-dataset.csv')
X = df.drop('pIC50', axis=1)
Y = df.pIC50

selection = VarianceThreshold(threshold=(.8 * (1 - .8)))
X = selection.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
np.random.seed(100)
model = RandomForestRegressor(n_estimators=200)
model.fit(X_train, Y_train)
r2 = model.score(X_test, Y_test)
Y_pred = model.predict(X_test)

# sns.set(color_codes=True)
# sns.set_style("white")
#
# ax = sns.regplot(Y_test, Y_pred, scatter_kws={'alpha':0.4})
# ax.set_xlabel('Experimental pIC50', fontsize='large', fontweight='bold')
# ax.set_ylabel('Predicted pIC50', fontsize='large', fontweight='bold')
# ax.set_xlim(0, 12)
# ax.set_ylim(0, 12)
# ax.figure.set_size_inches(5, 5)
# plt.show()