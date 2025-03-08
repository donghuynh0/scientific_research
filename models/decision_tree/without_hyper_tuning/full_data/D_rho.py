''' y = D * p '''

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from setup import load_data
from evaluate_model import evaluate

# load data
data = load_data()


X = data[['T*', 'rho*']]
y = data['D*'] * data['rho*']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(max_depth=5, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Create DataFrame for comparison
df = pd.DataFrame({'Actual D*rho': y_test, 'Predicted D*rho': y_pred})

# Compute absolute error and relative error
df['Absolute Error'] = np.abs(df['Actual D*rho'] - df['Predicted D*rho'])
df['Relative Error (%)'] = (df['Absolute Error'] / df['Actual D*rho']) * 100

evaluate(df, y_test, y_pred)