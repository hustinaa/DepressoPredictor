import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import os

from django.core.management.base import BaseCommand
from django.conf import settings

class Command(BaseCommand):
    help = 'Train KNN and Decision Tree models and save them'

    def handle(self, *args, **kwargs):
        df = pd.read_csv(os.path.join(settings.BASE_DIR, 'espressoapp', 'Student Mental health.csv'))
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

        year_mapping = {'year 1': 1, 'Year 1': 1, 'year 2': 2, 'Year 2': 2, 'year 3': 3, 'Year 3': 3, 'year 4': 4, 'Year 4': 4}
        df['Your current year of Study'] = df['Your current year of Study'].map(year_mapping)
        df.dropna()

        X = df.drop(columns=['Do you have Depression?'])
        y = df['Do you have Depression?']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        scalers = {}
        for col in X_train.columns:
            scaler = MinMaxScaler()
            X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
            scalers[col] = scaler

        for col in X_test.columns:
            X_test[col] = scalers[col].transform(X_test[col].values.reshape(-1, 1))

        # Print scaler keys for debugging
        print(f"Scaler keys: {scalers.keys()}")

        # Train KNN
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)
        knn_acc = accuracy_score(y_test, y_pred)
        self.stdout.write(self.style.SUCCESS('KNN Accuracy: {0}'.format(knn_acc)))

        # Save the KNN model and scalers
        joblib.dump(knn_model, os.path.join(settings.BASE_DIR, 'espressoapp', 'knn_model.pkl'))
        joblib.dump(scalers, os.path.join(settings.BASE_DIR, 'espressoapp', 'scalers.pkl'))

        # Train Decision Tree
        dt_model = DecisionTreeClassifier()
        dt_model.fit(X_train, y_train)
        y_pred = dt_model.predict(X_test)
        dt_acc = accuracy_score(y_test, y_pred)
        self.stdout.write(self.style.SUCCESS('Decision Tree Accuracy: {0}'.format(dt_acc)))

        # Save the Decision Tree model
        joblib.dump(dt_model, os.path.join(settings.BASE_DIR, 'espressoapp', 'decision_tree_model.pkl'))
