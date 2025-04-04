''' y = D * p '''

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from utils.setup import load_data
from utils.evaluate_model import evaluate

# load data
data = load_data()


X = data[['T*', 'rho*']]
y = data['D*'] * data['rho*']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(random_state=42)

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10, 20]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Create DataFrame for comparison
df = pd.DataFrame({'Actual D* * rho*': y_test, 'Predicted D* * rho*': y_pred})

# Compute absolute error and relative error
df['Absolute Error'] = np.abs(df['Actual D* * rho*'] - df['Predicted D* * rho*'])
df['Relative Error (%)'] = (df['Absolute Error'] / df['Actual D* * rho*']) * 100

evaluate(df, y_test, y_pred, grid_search)