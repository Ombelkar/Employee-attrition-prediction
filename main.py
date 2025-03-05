import pandas as pd  # Add this at the top

# Load the dataset
data = pd.read_csv("C:/Users/BAPS/Desktop/HR New Columns Excel.csv")

# Show basic info
print(data.info())

print(data.columns)

from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Label Encoding for binary variables
label_encoder = LabelEncoder()

# Encoding 'Attrition' and 'OverTime' to 0/1
data['Attrition'] = label_encoder.fit_transform(data['Attrition'])
data['OverTime'] = label_encoder.fit_transform(data['OverTime'])

# One-Hot Encoding for 'MaritalStatus'
data = pd.get_dummies(data, columns=['MaritalStatus'], drop_first=True)  # drop_first avoids dummy variable trap

# Check the transformed data
print(data.head())

from sklearn.preprocessing import StandardScaler

# Define numerical columns to scale
numerical_columns = ['Age', 'DistanceFromHome', 'YearsAtCompany', 'MonthlyIncome', 'WorkLifeBalance', 'NumCompaniesWorked']

# Initialize StandardScaler
scaler = StandardScaler()

# Scale the numerical features
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Check the transformed data
print(data.head())

from sklearn.model_selection import train_test_split

# Split the data into features (X) and target (y)
X = data.drop('Attrition', axis=1)  # All columns except 'Attrition'
y = data['Attrition']  # 'Attrition' column is the target variable

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape of the splits
print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest Classifier
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Check accuracy on training data
train_accuracy = model.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using accuracy
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Accuracy score
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Classification report (precision, recall, f1-score)
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

from sklearn.model_selection import GridSearchCV

# Define the parameters to tune
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Get the best parameters
print(f"Best Parameters: {grid_search.best_params_}")

# Use the best model to predict on the test set
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Evaluate the best model
print(f"Optimized Test Accuracy: {accuracy_score(y_test, y_pred_best) * 100:.2f}%")


