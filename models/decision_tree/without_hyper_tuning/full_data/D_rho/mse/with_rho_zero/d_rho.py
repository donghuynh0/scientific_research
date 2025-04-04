""" y = D * rho """

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from utils.setup import load_data
from utils.evaluate_model import evaluate
from utils.plot_scatter import plot_scatter, plot_relative_error
from utils.setup import synthetic_data
# load data
data = load_data()


X = data[['T*', 'rho*']]
y = data['D*'] * data['rho*']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create new data at rho is zero
synthetic_data_train = synthetic_data(X_train, 'D* x rho*')

# merge data
X_train_augmented = pd.concat([X_train, synthetic_data_train[['T*', 'rho*']]], ignore_index=True)
y_train_augmented = pd.concat([y_train, synthetic_data_train['D* x rho*']], ignore_index=True)

# train with MSE criterion
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train_augmented, y_train_augmented)

# predict
y_pred = model.predict(X_test)

df = pd.DataFrame({'Actual D*rho': y_test, 'Predicted D*rho': y_pred})

# compute errors
df['Absolute Error'] = np.abs(df['Actual D*rho'] - df['Predicted D*rho'])
df['Relative Error (%)'] = (df['Absolute Error'] / df['Actual D*rho']) * 100

evaluate(df, y_test, y_pred)
print(f"Max Depth of the trained model: {model.get_depth()}")


# plot
plot_scatter(data['rho*'], data['D* x rho*'], 'rho*', 'D* x rho*')
plot_relative_error(X_test, y_test, y_pred, 'D* x rho*')