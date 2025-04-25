import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from utils.calculations import dpt

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


def load_augmented_data():
    data = load_data()
    train_data, test_data = load_splited_data()
    new_data = train_data[['T*', 'rho*', 'D* x rho*', '(D* x rho*) / sqrt(T*)']]
    t_values = data['T*'].value_counts().index
    augmented_data_train = pd.DataFrame({
        'T*': t_values,
        'rho*': 0,
        'D* x rho*': [dpt(T) for T in t_values],
        '(D* x rho*) / sqrt(T*)': [dpt(T) / np.sqrt(T) for T in t_values]
    })

    new_train_data = pd.concat([new_data, augmented_data_train], ignore_index=True)

    return new_train_data, test_data
