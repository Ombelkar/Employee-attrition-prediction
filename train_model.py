import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
file_path = "C:/Users/BAPS/Desktop/HR New Columns Excel.csv"
df = pd.read_csv(file_path)

# Define features and target
features = ['Age', 'DistanceFromHome', 'YearsAtCompany', 'EnvironmentSatisfaction', 
            'JobSatisfaction', 'MonthlyIncome', 'WorkLifeBalance', 'NumCompaniesWorked']
target = 'Attrition'

X = df[features]
y = df[target].map({'Yes': 1, 'No': 0})  # Convert categorical target to binary

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully!")
