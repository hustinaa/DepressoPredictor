import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from django.conf import settings

def predict_with_knn(data):
    model_path = os.path.join(settings.BASE_DIR, 'espressoapp', 'knn_model.pkl')
    scaler_path = os.path.join(settings.BASE_DIR, 'espressoapp', 'scalers.pkl')
    
    model = joblib.load(model_path)
    scalers = joblib.load(scaler_path)
    
    # Print scaler keys for debugging
    print(f"Scaler keys: {scalers.keys()}")
    
    scaled_data = []
    feature_mapping = {
        'gender': 'Choose your gender',
        'age': 'Age',
        'marital_status': 'Marital status',
        'anxiety': 'Do you have Anxiety?',
        'panic_attack': 'Do you have Panic attack?',
        'specialist_treatment': 'Did you seek any specialist for a treatment?',
        'year_of_study': 'Your current year of Study'
    }
    
    for feature, mapped_feature in feature_mapping.items():
        if mapped_feature not in scalers:
            raise KeyError(f"Feature '{mapped_feature}' not found in scalers")
        scaled_feature = scalers[mapped_feature].transform(np.array(data[feature]).reshape(1, -1))
        scaled_data.append(scaled_feature[0])
    
    scaled_data = np.array(scaled_data).reshape(1, -1)
    prediction = model.predict(scaled_data)[0]  # Ensure the prediction is an integer
    return 1 if prediction == 'Yes' else 0

def predict_with_decision_tree(data):
    model_path = os.path.join(settings.BASE_DIR, 'espressoapp', 'decision_tree_model.pkl')
    scaler_path = os.path.join(settings.BASE_DIR, 'espressoapp', 'scalers.pkl')
    
    model = joblib.load(model_path)
    scalers = joblib.load(scaler_path)
    
    # Print scaler keys for debugging
    print(f"Scaler keys: {scalers.keys()}")
    
    scaled_data = []
    feature_mapping = {
        'gender': 'Choose your gender',
        'age': 'Age',
        'marital_status': 'Marital status',
        'anxiety': 'Do you have Anxiety?',
        'panic_attack': 'Do you have Panic attack?',
        'specialist_treatment': 'Did you seek any specialist for a treatment?',
        'year_of_study': 'Your current year of Study'
    }
    
    for feature, mapped_feature in feature_mapping.items():
        if mapped_feature not in scalers:
            raise KeyError(f"Feature '{mapped_feature}' not found in scalers")
        scaled_feature = scalers[mapped_feature].transform(np.array(data[feature]).reshape(1, -1))
        scaled_data.append(scaled_feature[0])
    
    scaled_data = np.array(scaled_data).reshape(1, -1)
    prediction = model.predict(scaled_data)[0]  # Ensure the prediction is an integer
    return 1 if prediction == 'Yes' else 0

def get_knn_accuracy():
    model_path = os.path.join(settings.BASE_DIR, 'espressoapp', 'knn_model.pkl')
    model = joblib.load(model_path)
    
    # Load and preprocess the dataset
    csv_path = os.path.join(settings.BASE_DIR, 'espressoapp', 'Student Mental health.csv')
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    
    columns_to_drop = ['Timestamp', 'What is your course?', 'What is your CGPA?']
    df.drop(columns=columns_to_drop, inplace=True)
    
    sex_mapping = {'Male': 0, 'Female': 1}
    df['Choose your gender'] = df['Choose your gender'].map(sex_mapping)
    
    marital_mapping = {'No': 0, 'Yes': 1}
    df['Marital status'] = df['Marital status'].map(marital_mapping)
    
    anxiety_mapping = {'No': 0, 'Yes': 1}
    df['Do you have Anxiety?'] = df['Do you have Anxiety?'].map(anxiety_mapping)
    
    panic_mapping = {'No': 0, 'Yes': 1}
    df['Do you have Panic attack?'] = df['Do you have Panic attack?'].map(panic_mapping)
    
    spec_mapping = {'No': 0, 'Yes': 1}
    df['Did you seek any specialist for a treatment?'] = df['Did you seek any specialist for a treatment?'].map(spec_mapping)
    
    year_mapping = {
        'year 1': 1, 'Year 1': 1, 'year 2': 2, 'Year 2': 2,
        'year 3': 3, 'Year 3': 3, 'year 4': 4, 'Year 4': 4
    }
    df['Your current year of Study'] = df['Your current year of Study'].map(year_mapping)
    
    X = df.drop(columns=['Do you have Depression?'])
    y = df['Do you have Depression?']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Load scalers
    scaler_path = os.path.join(settings.BASE_DIR, 'espressoapp', 'scalers.pkl')
    scalers = joblib.load(scaler_path)
    
    for col in X_test.columns:
        X_test[col] = scalers[col].transform(X_test[col].values.reshape(-1, 1))
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def get_decision_tree_accuracy():
    model_path = os.path.join(settings.BASE_DIR, 'espressoapp', 'decision_tree_model.pkl')
    model = joblib.load(model_path)
    
    # Load and preprocess the dataset
    csv_path = os.path.join(settings.BASE_DIR, 'espressoapp', 'Student Mental health.csv')
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    
    columns_to_drop = ['Timestamp', 'What is your course?', 'What is your CGPA?']
    df.drop(columns=columns_to_drop, inplace=True)
    
    sex_mapping = {'Male': 0, 'Female': 1}
    df['Choose your gender'] = df['Choose your gender'].map(sex_mapping)
    
    marital_mapping = {'No': 0, 'Yes': 1}
    df['Marital status'] = df['Marital status'].map(marital_mapping)
    
    anxiety_mapping = {'No': 0, 'Yes': 1}
    df['Do you have Anxiety?'] = df['Do you have Anxiety?'].map(anxiety_mapping)
    
    panic_mapping = {'No': 0, 'Yes': 1}
    df['Do you have Panic attack?'] = df['Do you have Panic attack?'].map(panic_mapping)
    
    spec_mapping = {'No': 0, 'Yes': 1}
    df['Did you seek any specialist for a treatment?'] = df['Did you seek any specialist for a treatment?'].map(spec_mapping)
    
    year_mapping = {
        'year 1': 1, 'Year 1': 1, 'year 2': 2, 'Year 2': 2,
        'year 3': 3, 'Year 3': 3, 'year 4': 4, 'Year 4': 4
    }
    df['Your current year of Study'] = df['Your current year of Study'].map(year_mapping)
    
    X = df.drop(columns=['Do you have Depression?'])
    y = df['Do you have Depression?']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Load scalers
    scaler_path = os.path.join(settings.BASE_DIR, 'espressoapp', 'scalers.pkl')
    scalers = joblib.load(scaler_path)
    
    for col in X_test.columns:
        X_test[col] = scalers[col].transform(X_test[col].values.reshape(-1, 1))
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy