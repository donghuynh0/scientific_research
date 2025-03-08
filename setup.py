import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

def load_data():
    filepath = os.getenv("FILE_PATH")
    try:
        data = pd.read_excel(filepath)
        data = data.dropna()
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