from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Load model
model_path = 'models/churn_model.pkl'
if not os.path.exists(model_path):
    print("Warning: Model not found. Run train.py to generate it.")
    model = None
else:
    model = joblib.load(model_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'Churn Prediction'}), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        
        # Prepare features
        features = prepare_features(data)
        features_array = pd.DataFrame([features])
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        
        return jsonify({
            'churn_probability': float(probability[1]),
            'prediction': 'churn' if prediction == 1 else 'no_churn',
            'confidence': float(max(probability)),
            'risk_level': get_risk_level(probability[1])
        }), 200
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        customers = data.get('customers', [])
        
        predictions = []
        for customer in customers:
            features = prepare_features(customer)
            features_array = pd.DataFrame([features])
            prediction = model.predict(features_array)[0]
            probability = model.predict_proba(features_array)[0]
            
            predictions.append({
                'customer_id': customer.get('customer_id'),
                'churn_probability': float(probability[1]),
                'prediction': 'churn' if prediction == 1 else 'no_churn',
                'risk_level': get_risk_level(probability[1])
            })
        
        return jsonify({'predictions': predictions}), 200
    
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    try:
        metrics_path = 'models/metrics.pkl'
        if os.path.exists(metrics_path):
            metrics = joblib.load(metrics_path)
            return jsonify(metrics), 200
        else:
            return jsonify({'error': 'Metrics not available'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/features', methods=['GET'])
def get_features():
    try:
        features_path = 'models/feature_importance.pkl'
        if os.path.exists(features_path):
            importance = joblib.load(features_path)
            return jsonify(importance), 200
        else:
            return jsonify({'error': 'Feature importance not available'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def prepare_features(data):
    """Prepare features for model"""
    return {
        'age': data.get('age', 0),
        'tenure': data.get('tenure', 0),
        'monthly_charges': data.get('monthly_charges', 0),
        'total_charges': data.get('total_charges', 0),
        'contract_month_to_month': 1 if data.get('contract_type') == 'month-to-month' else 0,
        'contract_one_year': 1 if data.get('contract_type') == 'one_year' else 0,
        'internet_service_fiber': 1 if data.get('internet_service') == 'fiber' else 0,
        'internet_service_dsl': 1 if data.get('internet_service') == 'dsl' else 0,
        'paperless_billing': 1 if data.get('paperless_billing') else 0,
        'phone_service': 1 if data.get('phone_service') else 0,
        'online_security': 1 if data.get('online_security') else 0,
        'tech_support': 1 if data.get('tech_support') else 0,
        'streaming_tv': 1 if data.get('streaming_tv') else 0,
        'streaming_movies': 1 if data.get('streaming_movies') else 0,
        'multiple_lines': 1 if data.get('multiple_lines') else 0
    }

def get_risk_level(probability):
    """Determine risk level"""
    if probability < 0.3:
        return 'low'
    elif probability < 0.6:
        return 'medium'
    else:
        return 'high'

if __name__ == '__main__':
    app.run(debug=True, port=5003)