# Employee Attrition Prediction

## Project Overview
This project aims to predict employee attrition using machine learning based on key factors such as age, salary, job satisfaction, and work-life balance. The model helps HR teams make data-driven decisions to improve employee retention.

## Features
- Built a predictive model to analyze employee attrition trends.
- Used a **correlation heatmap** to select the most important features.
- Implemented a **Flask web application** for user interaction.
- Provides an interactive interface where HR can input employee details and get predictions.

## Tech Stack
- **Python**: Data preprocessing, model training, deployment.
- **Pandas, NumPy**: Data manipulation.
- **Scikit-Learn**: Machine learning model.
- **Flask**: Web framework for deployment.
- **Matplotlib, Seaborn**: Data visualization.

## Dataset
The dataset includes employee-related attributes such as age, income, job satisfaction, and work-life balance.

**Dataset link:** https://www.kaggle.com/code/faressayah/ibm-hr-analytics-employee-attrition-performance
- **Raw Dataset:** `Employee_attrition_dataset.csv` (all features)
- **Preprocessed Dataset:** `HR_New_Columns_Excel.xlsx` (selected features based on correlation analysis)


## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Ombelkar/Employee-Attrition-Prediction.git
   cd Employee-Attrition-Prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask app:
   ```bash
   python app.py
   ```
4. Open the browser and go to:
   ```
   http://127.0.0.1:5000/
   ```
5. Enter employee details and get the prediction!

## Model Training
- **`train_model.py`**: Contains code for preprocessing, training, and saving the model.
- Uses selected features based on correlation analysis.
- Saves trained model and scaler as **`model.pkl`** and **`scalar.pkl`**.

## Flask Web App
- **`app.py`**: Loads `model.pkl` and `scalar.pkl` to make predictions.
- Provides a simple UI for HR teams to input employee details.

## Screenshots
Screenshots are available in the `screenshots/` folder.

## License
This project is licensed under the MIT License.

---
This README provides a structured overview of the project, guiding users on understanding and using it effectively. 🚀
