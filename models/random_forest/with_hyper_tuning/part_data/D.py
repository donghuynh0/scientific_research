''' y = D '''

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
y_train = train_data['D*']
X_test = test_data[['T*', 'rho*']]
y_test = test_data['D*']

model = RandomForestRegressor(random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],         # Number of trees
    'max_depth': [3, 5, 10, 15, None],      # Tree depth
    'min_samples_split': [2, 5, 10],        # Min samples to split
    'min_samples_leaf': [1, 2, 4]           # Min samples per leaf
}

# Perform GridSearchCV
grid_search = GridSearchCV(
    model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Predict using the best model
y_pred = best_model.predict(X_test)

# Create DataFrame for comparison
df = pd.DataFrame({'Actual D*': y_test, 'Predicted D*': y_pred})
df['Absolute Error'] = np.abs(df['Actual D*'] - df['Predicted D*'])
df['Relative Error (%)'] = (df['Absolute Error'] / df['Actual D*']) * 100

evaluate(df, y_test, y_pred, grid_search)
