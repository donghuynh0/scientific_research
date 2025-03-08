''' y = D '''

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
y_train = train_data['D*']
X_test = test_data[['T*', 'rho*']]
y_test = test_data['D*']

model = DecisionTreeRegressor(random_state=42)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

df = pd.DataFrame({'Actual D*': y_test, 'Predicted D*': y_pred})

df['Absolute Error'] = np.abs(df['Actual D*'] - df['Predicted D*'])
df['Relative Error (%)'] = (df['Absolute Error'] / df['Actual D*']) * 100

evaluate(df, y_test, y_pred)