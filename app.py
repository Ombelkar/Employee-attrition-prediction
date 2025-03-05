from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract 8 input features from form
        input_data = [float(request.form[col]) for col in ['Age', 'DistanceFromHome', 'YearsAtCompany', 
                                                           'EnvironmentSatisfaction', 'JobSatisfaction', 
                                                           'MonthlyIncome', 'WorkLifeBalance', 'NumCompaniesWorked']]
        
        # Scale input data
        input_scaled = scaler.transform([input_data])

        # Make prediction
        prediction = model.predict(input_scaled)
        result = "Employee will leave" if prediction[0] == 1 else "Employee will stay"

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
