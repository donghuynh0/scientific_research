import numpy as np
import pandas as pd
from utils.calculations import mae_weighted, mse_weighted, relative_error_weighted


def evaluate(y_train, y_train_pred, y_test, y_test_pred):
    def compute_metrics(df, phase):
        mse = (np.abs(df["Absolute Error"]) ** 2).mean()
        mae = df["Absolute Error"].mean()
        mre = df["Relative Error (%)"].mean()

        print(f"---- {phase} Performance ----")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Mean Relative Error: {mre:.2f}%\n")

    # Create DataFrame for training results
    df_train = pd.DataFrame({'Actual D*': y_train, 'Predicted D*': y_train_pred})
    df_train['Absolute Error'] = np.abs(df_train['Actual D*'] - df_train['Predicted D*'])
    df_train['Squared Error'] = (df_train['Actual D*'] - df_train['Predicted D*']) ** 2
    df_train['Relative Error (%)'] = (df_train['Absolute Error'] / df_train['Actual D*']) * 100

    # Create DataFrame for test results
    df_test = pd.DataFrame({'Actual D*': y_test, 'Predicted D*': y_test_pred})
    df_test['Absolute Error'] = np.abs(df_test['Actual D*'] - df_test['Predicted D*'])
    df_test['Squared Error'] = (df_test['Actual D*'] - df_test['Predicted D*']) ** 2
    df_test['Relative Error (%)'] = (df_test['Absolute Error'] / df_test['Actual D*']) * 100

    # Compute and print metrics
    compute_metrics(df_train, "Training")
    compute_metrics(df_test, "Testing")


def evaluate_with_weighted(y_train, y_train_pred, y_test, y_test_pred):
    def compute_metrics(df, phase):
        mse = (np.abs(df["Absolute Error"]) ** 2).mean()
        mae = df["Absolute Error"].mean()
        mre = df["Relative Error (%)"].mean()

        print(f"---- {phase} Performance ----")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Mean Relative Error: {mre:.2f}%\n")

    # Create DataFrame for training results
    df_train = pd.DataFrame({'Actual D*': y_train, 'Predicted D*': y_train_pred})
    df_train['Absolute Error'] = np.abs(df_train['Actual D*'] - df_train['Predicted D*'])
    df_train['Squared Error'] = (df_train['Actual D*'] - df_train['Predicted D*']) ** 2
    df_train['Relative Error (%)'] = (df_train['Absolute Error'] / df_train['Actual D*']) * 100

    # Create DataFrame for test results
    df_test = pd.DataFrame({'Actual D*': y_test, 'Predicted D*': y_test_pred})
    df_test['Absolute Error'] = np.abs(df_test['Actual D*'] - df_test['Predicted D*'])
    df_test['Squared Error'] = (df_test['Actual D*'] - df_test['Predicted D*']) ** 2
    df_test['Relative Error (%)'] = (df_test['Absolute Error'] / df_test['Actual D*']) * 100

    # Compute and print metrics
    compute_metrics(df_train, "Training")
    compute_metrics(df_test, "Testing")
