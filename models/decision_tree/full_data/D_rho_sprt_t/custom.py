""" y = D * p / sqrt(T)"""

import numpy as np
from sklearn.model_selection import train_test_split
from models.decision_tree.decision_tree_regressor import CustomDecisionTreeRegressor as DecisionTreeRegressor
from utils.setup import load_data
from utils.evaluate_model import evaluate
from utils.plot_scatter import plot_relative_error

# load data
data = load_data()


X = data[['T*', 'rho*']]
y = (data['D*'] * data['rho*']) / np.sqrt(data['T*'])

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Predict on training and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

evaluate(y_train=y_train, y_train_pred=y_train_pred, y_test=y_test, y_test_pred=y_test_pred)

# plot
plot_relative_error(X_test, y_test, y_test_pred, '(D* x rho*) / sqrt(T*)')
