''' y = D * p / sqrt(T)'''

import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from utils.setup import load_splited_data
from utils.evaluate_model import evaluate

# load data
train_data, test_data = load_splited_data()

X_train = train_data[['T*', 'rho*']]
y_train = (train_data['D*'] * train_data['rho*']) / np.sqrt(train_data['T*'])
X_test = test_data[['T*', 'rho*']]
y_test = (test_data['D*'] * test_data['rho*']) / np.sqrt(test_data['T*'])

model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Create DataFrame for comparison
df = pd.DataFrame({'Actual D*ρ*/√T*': y_test, 'Predicted D*ρ*/√T*': y_pred})

# Compute absolute error and relative error
df['Absolute Error'] = np.abs(df['Actual D*ρ*/√T*'] - df['Predicted D*ρ*/√T*'])
df['Relative Error (%)'] = (df['Absolute Error'] / df['Actual D*ρ*/√T*']) * 100

evaluate(df, y_test, y_pred)
