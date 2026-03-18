# AI Pricing Prediction API

A machine learning API that predicts optimal product pricing using Random Forest.

## What it does
- Takes product data as input (cost, demand score, competitor price)
- Returns predicted price, revenue, margin and confidence level
- Integrated with n8n automation pipeline

## Tech Stack
- Python + Flask
- Scikit-learn (Random Forest model)
- n8n (automation pipeline)
- AWS EC2 (deployment)

## API Endpoints

### Health Check
GET /health

### Price Prediction
POST /predict
{
  "product_name": "Wireless Headphones",
  "cost": 25,
  "demand_score": 8,
  "competitor_price": 59.99,
  "season": "holiday"
}

## Setup
pip install -r requirements.txt
python train_model.py
python app.py

## Author
Nitesh Nankani
