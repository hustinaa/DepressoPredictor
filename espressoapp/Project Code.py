import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np

# import warnings
# warnings.filterwarnings('ignore')

df = pd.read_csv('Student Mental health.csv')
df.dropna(inplace=True)
#print(df)

columns_to_drop = [
    'Timestamp',
    'What is your course?',
    'What is your CGPA?'
]

df.drop(columns=columns_to_drop, inplace=True)

# Map the "Choose your gender" column into 0 for male and 1 for female
sex_mapping = {'Male': 0, 'Female': 1}
df['Choose your gender'] = df['Choose your gender'].map(sex_mapping)

# Map the "Marital status" (No=0, Yes=1)
marital_mapping = {
    'No':0,
    'Yes':1
}
df['Marital status'] = df['Marital status'].map(marital_mapping)

# Map the "Do you have Anxiety?" (No=0, Yes=1)
anxiety_mapping = {
    'No':0,
    'Yes': 1
}
df['Do you have Anxiety?'] = df['Do you have Anxiety?'].map(anxiety_mapping)

# Map the "Do you have Panic attack?" (No=0, Yes=1)
panic_mapping = {
    'No':0,
    'Yes':1
}
df['Do you have Panic attack?'] = df['Do you have Panic attack?'].map(panic_mapping)

# Map the "Did you seek any specialist for a treatment?" (No=0, Yes=1)
spec_mapping = {
    'No':0,
    'Yes':1
}
df['Did you seek any specialist for a treatment?'] = df['Did you seek any specialist for a treatment?'].map(spec_mapping)

# Map the "Your current year of Study" (year 1=1, year 2=2, year 3=3, year 4=4)

year_mapping = {
    'year 1': 1,
    'Year 1': 1,
    'year 2': 2,
    'Year 2': 2,
    'year 3': 3,
    'Year 3': 3,
    'year 4': 4,
    'Year 4': 4
}
df['Your current year of Study'] = df['Your current year of Study'].map(year_mapping)
df.dropna()

# Get the training set

df.dropna()
X = df.drop(columns=['Do you have Depression?'])
y = df['Do you have Depression?']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# FOR KNN: We need to do scaling
scalers = {}
for col in X_train.columns:
  scaler = MinMaxScaler()
  X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
  scalers[col] = scaler

# Apply Scaling into the Test Set
for col in X_test.columns:
  X_test[col] = scalers[col].transform(X_test[col].values.reshape(-1, 1))

#print(X_train)

# Model Training and Evaluation

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print('KNN Accuracy: {0}'.format(acc))

# Save the model and scalers for future use
joblib.dump(model, 'knn_model.pkl')
joblib.dump(scalers, 'scalers.pkl')

# Function to predict depression
def predict_depression(datapoint: dict):
    model = joblib.load('knn_model.pkl')
    scalers = joblib.load('scalers.pkl')
    
    datapoint_list = []
    for col in X.columns:
        if col in datapoint:
            value = datapoint[col]
            scaled_value = scalers[col].transform(np.array([[value]])).flatten()[0]
            datapoint_list.append(scaled_value)
        else:
            raise ValueError(f"Missing value for column {col}")
    
    datapoint_array = np.array(datapoint_list).reshape(1, -1)
    prediction = model.predict(datapoint_array)[0]
    
    return prediction

# Example usage

sex = input('What is your sex assigned at birth (0=Male, 1=Female): ')
age = int(input('Age: '))
marital = input('What is your marital status (0=Single, 1=Married): ')
anxiety = input('Do you have anxiety (0=No, 1=Yes): ')
panic = input('Do you experience panic attacks (0=No, 1=Yes): ')
help = input('Did you seek any specialist for a treatment (0=No, 1=Yes): ')
year = int(input('What is your current year of study: '))
new_data = {
    'Choose your gender': sex,  # Male
    'Age': age,
    'Marital status': marital,  # No
    'Do you have Anxiety?': anxiety,  # Yes
    'Do you have Panic attack?': panic,  # No
    'Did you seek any specialist for a treatment?': help,  # No
    'Your current year of Study': year  # Year 3
}

prediction = predict_depression(new_data)
print('KNN Prediction:', prediction)





# DECISION TREE

# Model Training and Evaluation
model1 = DecisionTreeClassifier()
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print('Decision Tree Accuracy: {0}'.format(acc))

# Save the model
joblib.dump(model1, 'decision_tree_model.pkl')

# Function to predict depression using Decision Tree
def predict_depression(datapoint: dict):
    model = joblib.load('decision_tree_model.pkl')
    scalers = joblib.load('scalers.pkl')
    
    datapoint_list = []
    for col in X.columns:
        if col in datapoint:
            value = datapoint[col]
            scaled_value = scalers[col].transform(np.array([[value]])).flatten()[0]
            datapoint_list.append(scaled_value)
        else:
            raise ValueError(f"Missing value for column {col}")
    
    datapoint_array = np.array(datapoint_list).reshape(1, -1)
    prediction = model.predict(datapoint_array)[0]
    
    return prediction

# Example usage

prediction1 = predict_depression(new_data)
print('Decision Tree Prediction:', prediction1)
