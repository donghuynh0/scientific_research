''' y = D * p '''

import sys
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from utils.setup import load_splited_data
from utils.evaluate_model import evaluate

# load data
train_data, test_data = load_splited_data()


X_train = train_data[['T*', 'rho*']]
y_train = train_data['D*'] * train_data['rho*']
X_test = test_data[['T*', 'rho*']]
y_test = test_data['D*'] * test_data['rho*']

model = DecisionTreeRegressor(random_state=42)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

df = pd.DataFrame({'Actual D*ρ*': y_test, 'Predicted D*ρ*': y_pred})

# Compute absolute and relative errors
df['Absolute Error'] = np.abs(df['Actual D*ρ*'] - df['Predicted D*ρ*'])
df['Relative Error (%)'] = (df['Absolute Error'] / df['Actual D*ρ*']) * 100

evaluate(df, y_test, y_pred)