from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# ============================================
# TRAIN MODEL IF IT DOESN'T EXIST
# ============================================
def train_and_save_model():
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np

    np.random.seed(42)
    n_samples = 500

    cost = np.random.uniform(5, 100, n_samples)
    demand_score = np.random.randint(1, 11, n_samples)
    competitor_price = np.random.uniform(20, 300, n_samples)
    season = np.random.choice([0, 1, 2], n_samples)

    X = np.column_stack([cost, demand_score, competitor_price, season])
    y = (cost * 1.3 + demand_score * 2.5 +
         competitor_price * 0.3 + season * 3 +
         np.random.normal(0, 2, n_samples))

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved!")
    return model

# Load or train model
if os.path.exists("model.pkl"):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded from file!")
else:
    print("No model found — training now...")
    model = train_and_save_model()

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

    features = np.array([[cost, demand_score, competitor_price, season]])
    predicted_price = round(float(model.predict(features)[0]), 2)
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
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
