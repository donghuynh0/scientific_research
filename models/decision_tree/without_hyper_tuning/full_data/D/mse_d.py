""" y = D """

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from utils.setup import load_data
from utils.evaluate_model import evaluate
from utils.plot_scatter import plot_scatter, plot_relative_error

# load data
data = load_data()

X = data[['T*', 'rho*']]
y = data['D*']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train with MSE criterion
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

df = pd.DataFrame({'Actual D*': y_test, 'Predicted D*': y_pred},)

# compute errors
df['Absolute Error'] = np.abs(df['Actual D*'] - df['Predicted D*'])
df['Relative Error (%)'] = (df['Absolute Error'] / df['Actual D*']) * 100

evaluate(df, y_test, y_pred)
print(f"Max Depth of the trained model: {model.get_depth()}")

# plot
plot_scatter(data['rho*'], data['D*'], 'rho*', 'D*')
plot_relative_error(X_test, y_test, y_pred)