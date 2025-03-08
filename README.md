# **Scientific Research Project**

## **Model Selection**

We use the following regression models for training and evaluation:  

- **Decision Tree Regressor**  
- **Random Forest Regressor**  

### **Target Variables**  
The models are trained and evaluated on three different target variables:  

1. **D**  
2. **D * ρ**  
3. **D * ρ / √T**  

---

## **Installation**

1. **Clone the repository:**
   ```sh
   git clone https://github.com/donghuynh0/scientific_research.git
   cd scientific_research
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

3. **Install the required packages:**
   ```sh
   pip install -r requirements.txt
   ```

---

## **Configuration**
Ensure the `.env` file is correctly set up with the path to your data file:
```sh
FILE_PATH=/path/to/your/data.xlsx
```

---

## **Usage**
To run the script, use:
```sh
python3 models/decision_tree/with_hyper_tuning/full_data/predict_D_star.py
```
Change the path to run different models.

---

## **Files and Directory Structure**
```
scientific_research/
│-- .env                    # Contains environment variables, including the data file path
│-- .gitignore               # Specifies files and directories to be ignored by Git
│-- requirements.txt         # Lists the required Python packages
│-- evaluate_model.py        # Contains functions to compute and print evaluation metrics
│-- setup.py                 # Contains functions to load and preprocess the data
│-- models/                  # Directory containing model training scripts
│   ├── decision_tree/       # Scripts related to Decision Tree models
│   ├── random_forest/       # Scripts related to Random Forest models
```

---

## **Acknowledgments**
This project utilizes the following libraries:  

- [scikit-learn](https://scikit-learn.org/)  
- [pandas](https://pandas.pydata.org/)  
- [numpy](https://numpy.org/)  
- [python-dotenv](https://github.com/theskumar/python-dotenv)  



