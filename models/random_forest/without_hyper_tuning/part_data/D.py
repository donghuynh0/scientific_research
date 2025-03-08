''' y = D '''

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from setup import load_splited_data
from evaluate_model import evaluate

# load data
train_data, test_data = load_splited_data()

X_train = train_data[['T*', 'rho*']]
y_train = train_data['D*']
X_test = test_data[['T*', 'rho*']]
y_test = test_data['D*']

model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Create DataFrame for comparison
df = pd.DataFrame({'Actual D*': y_test, 'Predicted D*': y_pred})

# Compute absolute error and relative error
df['Absolute Error'] = np.abs(df['Actual D*'] - df['Predicted D*'])
df['Relative Error (%)'] = (df['Absolute Error'] / df['Actual D*']) * 100

evaluate(df, y_test, y_pred)