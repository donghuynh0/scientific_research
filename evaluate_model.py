from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

def evaluate(df, y_test, y_pred, grid_search = None):
    # Compute metrics
    mse = mean_squared_error(y_test, y_pred)
    mean_absolute_error = df["Absolute Error"].mean()
    mean_relative_error = df["Relative Error (%)"].mean()

    # Print results
    if grid_search:
        print(f'Best Parameters: {grid_search.best_params_}')
    print(f'Mean Squared Error: {mse:.4f}')
    print(f'Mean Absolute Error: {mean_absolute_error:.4f}')
    print(f'Mean Relative Error: {mean_relative_error:.2f}%')
