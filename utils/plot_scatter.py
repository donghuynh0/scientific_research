import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_relative_error(X_test, y_test, y_pred, target_name="D*"):
    df = pd.DataFrame({
        'T*': X_test['T*'].values,
        'rho*': X_test['rho*'].values,
        f'Actual {target_name}': np.round(y_test, 3),
        f'Predicted {target_name}': np.round(y_pred, 3)
    })

    df['Absolute Error'] = np.abs(df[f'Actual {target_name}'] - df[f'Predicted {target_name}'])
    df['Relative Error (%)'] = np.round((df['Absolute Error'] / df[f'Actual {target_name}']) * 100, 3)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['rho*'], df['Relative Error (%)'], c=df['T*'], cmap='viridis', edgecolors='k', alpha=0.8)
    plt.colorbar(scatter, label='T*')
    plt.xlabel('ρ* (rho)')
    plt.ylabel('Relative Error (%)')
    plt.title(f"Relative Error vs. ρ* (rho) with target = {target_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

