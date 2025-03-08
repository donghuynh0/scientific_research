''' y = D * p / sqrt(T)'''

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from setup import load_splited_data
from evaluate_model import evaluate

# load data
train_data, test_data = load_splited_data()

X_train = train_data[['T*', 'rho*']]
y_train = (train_data['D*'] * train_data['rho*']) / np.sqrt(train_data['T*'])
X_test = test_data[['T*', 'rho*']]
y_test = (test_data['D*'] * test_data['rho*']) / np.sqrt(test_data['T*'])

model = DecisionTreeRegressor(random_state=42)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Create DataFrame for comparison
df = pd.DataFrame({'Actual D*ρ*/sqrt(T*)': y_test, 'Predicted D*ρ*/sqrt(T*)': y_pred})

# Compute absolute and relative errors
df['Absolute Error'] = np.abs(df['Actual D*ρ*/sqrt(T*)'] - df['Predicted D*ρ*/sqrt(T*)'])
df['Relative Error (%)'] = (df['Absolute Error'] / df['Actual D*ρ*/sqrt(T*)']) * 100

evaluate(df, y_test, y_pred)
