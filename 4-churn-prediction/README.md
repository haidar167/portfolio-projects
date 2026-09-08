# Customer Churn Prediction

Machine learning model to predict customer churn using scikit-learn, with a Flask API for real-time predictions.

## Features
- 📊 Trained ML model (scikit-learn)
- 🎯 Churn probability prediction
- 📈 Feature importance analysis
- 🔄 Model retraining capability
- ⚡ Real-time API predictions
- 📉 Performance metrics

## Tech Stack
- Python 3.9+
- scikit-learn
- Flask
- Pandas
- Joblib

## Setup

```bash
cd 4-churn-prediction
pip install -r requirements.txt
python app.py
```

Server runs on `http://localhost:5003`

## Model Training

If model file missing:
```bash
python train.py
```

## API Endpoints

### POST /api/predict
Predict churn for a customer.

**Request:**
```json
{
  "age": 35,
  "tenure": 12,
  "monthly_charges": 65.5,
  "total_charges": 1234.50,
  "contract_type": "month-to-month"
}
```

**Response:**
```json
{
  "churn_probability": 0.72,
  "prediction": "churn",
  "confidence": 0.95
}
```

### GET /api/metrics
Get model performance metrics.

### GET /api/features
Get feature importance.

## Model Details

- **Algorithm**: Gradient Boosting Classifier
- **Features**: 15 customer attributes
- **Accuracy**: ~85%
- **AUC-ROC**: 0.89

## License
MIT