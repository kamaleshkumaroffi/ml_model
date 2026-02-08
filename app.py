import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Load the trained model
model = joblib.load('logistic_regression_model.joblib')

# Instantiate the Flask application
app = Flask(__name__)

# Define the /predict endpoint with its full logic
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid JSON data provided"}), 400

    try:
        # Convert the received JSON data into a Pandas DataFrame
        if isinstance(data, dict):
            df_predict = pd.DataFrame([data])
        elif isinstance(data, list):
            df_predict = pd.DataFrame(data)
        else:
            return jsonify({"error": "JSON data must be a dictionary or a list of dictionaries"}), 400

        # Ensure columns match the training features
        expected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        if not all(col in df_predict.columns for col in expected_features):
            missing_features = [col for col in expected_features if col not in df_predict.columns]
            return jsonify({"error": f"Missing one or more expected features in the input data: {', '.join(missing_features)}"}), 400

        # Reorder columns to match the training order
        df_predict = df_predict[expected_features]

        # Make predictions using the loaded model
        predictions = model.predict(df_predict)

        # Convert predictions to a list for JSON serialization
        prediction_list = predictions.tolist()

        return jsonify({"predictions": prediction_list}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

print("Flask application code written to app.py")
