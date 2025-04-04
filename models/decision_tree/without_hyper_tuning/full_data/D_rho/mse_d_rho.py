""" y = D * rho """

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from setup import load_data
from evaluate_model import evaluate
from plot_scatter import plot_scatter, plot_relative_error

# load data
data = load_data()


X = data[['T*', 'rho*']]
y = data['D*'] * data['rho*']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train with MSE criterion
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

df = pd.DataFrame({'Actual D*rho': y_test, 'Predicted D*rho': y_pred})

# compute errors
df['Absolute Error'] = np.abs(df['Actual D*rho'] - df['Predicted D*rho'])
df['Relative Error (%)'] = (df['Absolute Error'] / df['Actual D*rho']) * 100

evaluate(df, y_test, y_pred)


# plot
plot_scatter(data['rho*'], data['D* x rho*'], 'rho*', 'D* x rho*')
plot_relative_error(X_test, y_test, y_pred, 'D* x rho*')