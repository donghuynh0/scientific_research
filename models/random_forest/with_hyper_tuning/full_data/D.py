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

from setup import load_data
from evaluate_model import evaluate

# load data
data = load_data()

X = data[['T*', 'rho*']]
y = data['D*']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

# Create DataFrame for comparison
df = pd.DataFrame({'Actual D*': y_test, 'Predicted D*': y_pred})
df['Absolute Error'] = np.abs(df['Actual D*'] - df['Predicted D*'])
df['Relative Error (%)'] = (df['Absolute Error'] / df['Actual D*']) * 100

evaluate(df, y_test, y_pred, grid_search)