''' y = D * p / sqrt(T)'''

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from setup import load_splited_data
from evaluate_model import evaluate

# load data
train_data, test_data = load_splited_data()

X_train = train_data[['T*', 'rho*']]
y_train = (train_data['D*'] * train_data['rho*']) / np.sqrt(train_data['T*'])
X_test = test_data[['T*', 'rho*']]
y_test = (test_data['D*'] * test_data['rho*']) / np.sqrt(test_data['T*'])

model = RandomForestRegressor(random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],         # Number of trees
    'max_depth': [3, 5, 10, 15, None],      # Tree depth
    'min_samples_split': [2, 5, 10],        # Min samples to split
    'min_samples_leaf': [1, 2, 4]           # Min samples per leaf
}

grid_search = GridSearchCV(
    model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

# Create DataFrame for comparison
df = pd.DataFrame({'Actual (D* * rho* / sqrt(T*))': y_test, 'Predicted (D* * rho* / sqrt(T*))': y_pred})
df['Absolute Error'] = np.abs(df['Actual (D* * rho* / sqrt(T*))'] - df['Predicted (D* * rho* / sqrt(T*))'])
df['Relative Error (%)'] = (df['Absolute Error'] / df['Actual (D* * rho* / sqrt(T*))']) * 100

evaluate(df, y_test, y_pred, grid_search)