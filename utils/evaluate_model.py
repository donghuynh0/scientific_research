import numpy as np
import pandas as pd
from utils.calculations import mae_weighted, mse_weighted, relative_error_weighted


def evaluate(y_train, y_train_pred, y_test, y_test_pred, flag=True):
    def compute_metrics(df, phase, flag=True):
        mse = (np.abs(df["Absolute Error"]) ** 2).mean()
        mae = df["Absolute Error"].mean()
        mre = df["Relative Error (%)"].mean()
        if flag:
            print(f"---- {phase} Performance ----")
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"Mean Absolute Error: {mae:.4f}")
            print(f"Mean Relative Error: {mre:.2f}%\n")

        return round(mse, 4), round(mae, 4), round(mre, 2)

    # Training results
    df_train = pd.DataFrame({'Actual D*': y_train, 'Predicted D*': y_train_pred})
    df_train['Absolute Error'] = np.abs(df_train['Actual D*'] - df_train['Predicted D*'])
    df_train['Relative Error (%)'] = (df_train['Absolute Error'] / np.maximum(np.abs(df_train['Actual D*']), 1e-8)) * 100
    train_mse, train_mae, train_re = compute_metrics(df_train, "Training", flag)

    # Test results
    df_test = pd.DataFrame({'Actual D*': y_test, 'Predicted D*': y_test_pred})
    df_test['Absolute Error'] = np.abs(df_test['Actual D*'] - df_test['Predicted D*'])
    df_test['Relative Error (%)'] = (df_test['Absolute Error'] / np.maximum(np.abs(df_test['Actual D*']), 1e-8)) * 100
    test_mse, test_mae, test_re = compute_metrics(df_test, "Testing", flag)

    return {
        "train_MSE": train_mse,
        "train_MAE": train_mae,
        "train_RE": train_re,
        "test_MSE": test_mse,
        "test_MAE": test_mae,
        "test_RE": test_re
    }


