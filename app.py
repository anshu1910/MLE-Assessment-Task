from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask App
app = Flask(__name__)

# Load the trained model
model = joblib.load("models/house_price_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint to predict house price based on input features.
    Expected input: JSON {"features": [val1, val2, val3, ...]}
    """
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"predicted_price": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
