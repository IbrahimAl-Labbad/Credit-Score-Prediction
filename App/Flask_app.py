from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("C:\\Users\\AcTivE\\Desktop\\Project\\CreditScorePrediction\\App\\best_random_forest_model.joblib" )

# Define the API route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON data
        input_data = request.get_json()

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make predictions
        predictions = model.predict(input_df)

        # Return predictions as a JSON response
        return jsonify({
            'success': True,
            'prediction': int(predictions[0])  # Convert NumPy type to native Python
        })

    except Exception as e:
        # Return error message
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
