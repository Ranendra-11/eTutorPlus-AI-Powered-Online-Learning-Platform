import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("etutor_data.csv")  # Replace with actual dataset path

# Select relevant features and target variable
features = ["engagement_score", "time_spent", "quiz_scores", "ai_assist_usage"]
target = "student_success"

# Handle missing values
df = df.dropna()

# Convert target variable into binary classification
df[target] = (df[target] > df[target].median()).astype(int)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print(f"Random Forest Model Accuracy: {rf_accuracy * 100:.2f}%")

# Function to predict student success
def predict_success(student_features, model=rf_model):
    student_features = scaler.transform([student_features])
    prediction = model.predict(student_features)
    return "Likely to Succeed" if prediction[0] == 1 else "Needs Assistance"

# Example prediction
sample_student = df[features].iloc[0].values
print("Prediction for sample student:", predict_success(sample_student))

