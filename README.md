# Electric Motor Temperature Prediction

## 1. Project Overview
The goal of this project was to develop a machine learning model to predict the rotor temperature of a permanent-magnet synchronous machine (PMSM) based on various sensor readings. This is a regression task.

## 2. Data Preprocessing
The dataset was loaded and pre-processed. The 'profile_id' and 'torque' columns were dropped as they were not needed for prediction. The data was checked for null values, and features were scaled using StandardScaler to prepare them for model training.

## 3. Model Building and Evaluation
Four different regression models were trained and evaluated:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Machine (SVR)

The models were evaluated using R-squared (R²) and Root Mean Squared Error (RMSE). The Random Forest Regressor was the best-performing model with an R² score of **0.9995** and an RMSE of **0.43**.

## 4. Final Application
A web application was built using the Flask framework to serve the trained Random Forest model. The application provides a web form where a user can input the 10 required feature values and get a real-time prediction of the motor's rotor temperature.

## 5. How to Run the Application
1. Clone this repository.
2. Install the required libraries: `pip install -r requirements.txt`
3. Run the Flask application: `python app.py`
4. Open your web browser and go to `http://127.0.0.1:5000`.

## Model File Notice

The full trained model (`model.pkl`) is too large for GitHub (9.2GB).  
You can download it via this external link:  
[Download model.pkl (Google Drive)]([https://drive.google.com/file/d/1E7QE4DnfLTgDdPzRamQoTqvqIyuChIU9/view?usp=sharing})

Alternatively, a small demonstration model trained on a subset of the data is available in this repository for reviewers to test the application workflow.

---
