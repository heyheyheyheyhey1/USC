import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold

from sklearn.metrics import mean_squared_error, r2_score
import pickle

df = pd.read_csv('data/sars-3c-like-dataset.csv')
X = df.drop('pIC50', axis=1)
Y = df.iloc[:,-1]

def remove_low_variance(input_data, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]

X = remove_low_variance(X, threshold=0.1)
X.to_csv('descriptor_list.csv', index = False)

model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X, Y)
r2 = model.score(X, Y)
Y_pred = model.predict(X)
print('Mean squared error (MSE): %.2f'% mean_squared_error(Y, Y_pred))
print('Coefficient of determination (R^2): %.2f'% r2_score(Y, Y_pred))
pickle.dump(model, open('bioactivity-prediction-app-main/sars-3c-like-model.pkl', 'wb'))