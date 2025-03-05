import pickle
import numpy as np

# Load the trained model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the scaler
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Function to make predictions
def predict_attrition(input_data):
    input_data = np.array(input_data).reshape(1, -1)  # Reshape for model
    scaled_data = scaler.transform(input_data)  # Scale input data
    prediction = model.predict(scaled_data)  # Get prediction
    return "Will Leave" if prediction[0] == 1 else "Will Stay"

# Example usage
if __name__ == "__main__":
    sample_input = [35, 5000, 3, 2]  # Example input (age, salary, etc.)
    result = predict_attrition(sample_input)
    print("Attrition Prediction:", result)
