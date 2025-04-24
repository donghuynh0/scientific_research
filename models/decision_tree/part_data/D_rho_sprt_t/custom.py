""" y = D * p / sqrt(T)"""

import numpy as np
from sklearn.model_selection import train_test_split
from models.decision_tree.decision_tree_regressor import CustomDecisionTreeRegressor as DecisionTreeRegressor
from utils.setup import load_splited_data
from utils.evaluate_model import evaluate
from utils.plot_scatter import plot_relative_error

# load data
train_data, test_data = load_splited_data()

X_train = train_data[['T*', 'rho*']]
y_train = (train_data['D*'] * train_data['rho*']) / np.sqrt(train_data['T*'])
X_test = test_data[['T*', 'rho*']]
y_test = (test_data['D*'] * test_data['rho*']) / np.sqrt(test_data['T*'])

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Predict on training and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

evaluate(y_train=y_train, y_train_pred=y_train_pred, y_test=y_test, y_test_pred=y_test_pred)

# plot
plot_relative_error(X_test, y_test, y_test_pred, '(D* x rho*) / sqrt(T*)')
