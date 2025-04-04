''' y = D * p / sqrt(T)'''

import sys
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from utils.setup import load_splited_data
from utils.evaluate_model import evaluate

# load data
train_data, test_data = load_splited_data()

X_train = train_data[['T*', 'rho*']]
y_train = (train_data['D*'] * train_data['rho*']) / np.sqrt(train_data['T*'])
X_test = test_data[['T*', 'rho*']]
y_test = (test_data['D*'] * test_data['rho*']) / np.sqrt(test_data['T*'])

# Define Decision Tree model
model = DecisionTreeRegressor(random_state=42)

# Define hyperparameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20]
}

# Perform GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_

# Predict using the best model
y_pred = best_model.predict(X_test)

# Create DataFrame for comparison
df = pd.DataFrame({'Actual D* * rho* / sqrt(T*)': y_test, 'Predicted D* * rho* / sqrt(T*)': y_pred})
df['Absolute Error'] = np.abs(df['Actual D* * rho* / sqrt(T*)'] - df['Predicted D* * rho* / sqrt(T*)'])
df['Relative Error (%)'] = (df['Absolute Error'] / df['Actual D* * rho* / sqrt(T*)']) * 100

evaluate(df, y_test, y_pred, grid_search)