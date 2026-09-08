import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import joblib
import os

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Generate sample data
np.random.seed(42)
n_samples = 1000

data = {
    'age': np.random.randint(18, 80, n_samples),
    'tenure': np.random.randint(0, 72, n_samples),
    'monthly_charges': np.random.uniform(20, 120, n_samples),
    'total_charges': np.random.uniform(100, 8500, n_samples),
    'contract_month_to_month': np.random.randint(0, 2, n_samples),
    'contract_one_year': np.random.randint(0, 2, n_samples),
    'internet_service_fiber': np.random.randint(0, 2, n_samples),
    'internet_service_dsl': np.random.randint(0, 2, n_samples),
    'paperless_billing': np.random.randint(0, 2, n_samples),
    'phone_service': np.random.randint(0, 2, n_samples),
    'online_security': np.random.randint(0, 2, n_samples),
    'tech_support': np.random.randint(0, 2, n_samples),
    'streaming_tv': np.random.randint(0, 2, n_samples),
    'streaming_movies': np.random.randint(0, 2, n_samples),
    'multiple_lines': np.random.randint(0, 2, n_samples)
}

# Create target (churn) with some correlation to features
churn = ((data['tenure'] < 12) * 0.5 + 
         (data['contract_month_to_month'] * 0.3) + 
         (data['monthly_charges'] > 80) * 0.2 + 
         np.random.random(n_samples) * 0.1 > 0.5).astype(int)

X = pd.DataFrame(data)
y = pd.Series(churn, name='churn')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("Training model...")
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# Save model
joblib.dump(model, 'models/churn_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Save metrics
metrics = {
    'accuracy': float(accuracy),
    'auc_roc': float(auc_roc),
    'total_samples': len(X),
    'train_samples': len(X_train),
    'test_samples': len(X_test)
}
joblib.dump(metrics, 'models/metrics.pkl')

# Save feature importance
feature_importance = dict(zip(X.columns, model.feature_importances_))
joblib.dump(feature_importance, 'models/feature_importance.pkl')

print("\nModel saved to models/churn_model.pkl")
print(f"Feature Importance: {feature_importance}")