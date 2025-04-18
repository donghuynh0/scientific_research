import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from utils.calculations import d_rho

load_dotenv()


def load_data():
    filepath = os.getenv("FILE_PATH")
    try:
        data = pd.read_excel(filepath)
        data = data.dropna()
        data['D* x rho*'] = data['D*'] * data['rho*']
        data['(D* x rho*) / sqrt(T*)'] = data['D* x rho*'] / np.sqrt(data['T*'])
        return data
    except FileNotFoundError:
        print(f"Error: File not found")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def load_splited_data():
    data = load_data()
    test_data = data[data['rho*'] < 0.1]
    train_data = data[data['rho*'] >= 0.1]
    return train_data, test_data


def synthetic_data(X_train, target):
    t_values = X_train['T*'].value_counts().index

    if target == 'D* x rho*':
        rho_values = [d_rho(T) for T in t_values]
    elif target == '(D* x rho*) / sqrt(T*)':
        rho_values = [d_rho(T) / np.sqrt(T) for T in t_values]
    else:
        raise ValueError("Unknown target value")

    synthetic_data_train = pd.DataFrame({
        'T*': t_values,
        'rho*': rho_values,
        target: rho_values
    })

    return synthetic_data_train
