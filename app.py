from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# ============================================
# LOAD THE TRAINED MODEL WHEN FLASK STARTS
# ============================================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

print("ML Model loaded successfully!")

# ============================================
# SEASON MAPPING
# ============================================
season_map = {
    "holiday": 1,
    "new_year": 2,
    "regular": 0
}

def predict_price(product):
    cost = float(product["cost"])
    demand_score = float(product["demand_score"])
    competitor_price = float(product["competitor_price"])
    season = season_map.get(product.get("season", "regular"), 0)

    # Feed data into ML model
    features = np.array([[cost, demand_score, competitor_price, season]])
    predicted_price = round(float(model.predict(features)[0]), 2)

    # Make sure we don't price below cost
    predicted_price = max(predicted_price, cost * 1.1)

    estimated_units = demand_score * 120
    predicted_revenue = round(predicted_price * estimated_units, 2)
    margin = round(((predicted_price - cost) / predicted_price) * 100)

    if demand_score >= 7:
        confidence = "high"
    elif demand_score >= 4:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "product_name": product["product_name"],
        "predicted_price": predicted_price,
        "predicted_revenue": predicted_revenue,
        "estimated_units": estimated_units,
        "margin_percent": margin,
        "confidence": confidence,
        "model": "random_forest_v1"
    }

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("RECEIVED:", data)
        if not data:
            return jsonify({"error": "No data received"}), 400
        result = predict_price(data)
        print("RESULT:", result)
        return jsonify(result), 200
    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running", "model": "random_forest_v1"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)