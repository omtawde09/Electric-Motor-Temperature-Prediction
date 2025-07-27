from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and scaler
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: model.pkl or scaler.pkl not found. Make sure the files are in the root directory.")
    model, scaler = None, None


@app.route('/')
def home():
    # Render the main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return render_template('index.html', prediction_text='Error: Model or scaler not loaded.')

    try:
        # Get the values from the form and convert them to float
        # The order of features MUST match the order used during training
        features = [float(x) for x in request.form.values()]

        # The original feature order was:
        # ['u_q', 'coolant', 'stator_winding', 'u_d', 'stator_tooth',
        #  'motor_speed', 'i_d', 'i_q', 'stator_yoke', 'ambient']
        # We need to map form fields to this order
        form_data = request.form
        ordered_features = [
            float(form_data['u_q']),
            float(form_data['coolant']),
            float(form_data['stator_winding']),
            float(form_data['u_d']),
            float(form_data['stator_tooth']),
            float(form_data['motor_speed']),
            float(form_data['i_d']),
            float(form_data['i_q']),
            float(form_data['stator_yoke']),
            float(form_data['ambient'])
        ]

        # Convert to a numpy array and reshape for a single prediction
        final_features = np.array(ordered_features).reshape(1, -1)

        # Scale the features using the loaded scaler
        scaled_features = scaler.transform(final_features)

        # Make a prediction
        prediction = model.predict(scaled_features)

        # Format the output
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'Predicted Rotor Temperature: {output} °C')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error processing request: {e}')


if __name__ == "__main__":
    app.run(debug=True)
