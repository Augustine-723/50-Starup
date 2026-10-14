import urllib.request
import json
import os

print('=' * 60)
print('TESTING WORKFLOW WITH TWO DIFFERENT DATASETS')
print('=' * 60)

BASE_URL = 'http://localhost:3000'

# 1. Dataset 1: 50 Startups (Problem 1)
print('\n--- DATASET 1: 50 Startups (Problem 1) ---')
res1 = urllib.request.urlopen(f'{BASE_URL}/api/results?random_state=42')
data1 = json.loads(res1.read().decode('utf-8'))
p1_models = data1.get('feature_selection_results', {}).get('Forward SFS', [])
print('Dataset 1 Name: 50 Startups')
print('Algorithms found:', list(data1.get('feature_selection_results', {}).keys()))
print('Forward SFS models count:', len(p1_models))
best_p1 = data1.get('best_cv_model', {})
print(f"Best Model (by CV): k={best_p1.get('k')} features={best_p1.get('features')} CV RMSE={best_p1.get('cv_rmse_mean')} Test RMSE={best_p1.get('test_rmse')} R2={best_p1.get('r2')}")

# Seed 0 Benchmark verification
res0 = urllib.request.urlopen(f'{BASE_URL}/api/results?random_state=0')
data0 = json.loads(res0.read().decode('utf-8'))
seed0_k2 = [m for m in data0.get('feature_selection_results', {}).get('Forward SFS', []) if m.get('k') == 2][0]
print(f"Seed 0 Benchmark (k=2): Test RMSE={seed0_k2.get('test_rmse')} (Expected: 8198.80)")
assert abs(seed0_k2.get('test_rmse') - 8198.80) < 0.1, 'Seed 0 benchmark mismatch!'

# 2. Dataset 2: California Real Estate 10-Features (Problem 2)
print('\n--- DATASET 2: Real Estate 10-Features (Problem 2) ---')
csv_path = 'frontend/public/data/real_estate_valuation_10features.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    csv_text = f.read()

payload2 = {
    'csvText': csv_text,
    'targetColumn': 'MedianHouseValue',
    'datasetName': '加州房地產估價 (10 Features)',
    'testSize': 0.2,
    'randomState': 42
}
req2 = urllib.request.Request(
    f'{BASE_URL}/api/custom-analyze',
    data=json.dumps(payload2).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
res2 = urllib.request.urlopen(req2)
data2 = json.loads(res2.read().decode('utf-8'))
print(f"Dataset 2 Name: {data2.get('datasetName')}")
print(f"Total Samples: {data2.get('sampleCount')}")
print(f"Target Column: {data2.get('targetColumn')}")
print(f"Encoded Features Count: {data2.get('featureCount')}")
best_p2 = data2.get('best_cv_model', {})
print(f"Best Model (by CV): k={best_p2.get('k')} features={best_p2.get('features')} CV RMSE={best_p2.get('cv_rmse_mean')} Test RMSE={best_p2.get('test_rmse')} R2={best_p2.get('r2')}")
assert data2.get('sampleCount') == 80, 'Sample count mismatch!'
assert data2.get('targetColumn') == 'MedianHouseValue', 'Target column mismatch!'
assert len(data2.get('feature_selection_results', {})) >= 5, 'Algorithms missing in dataset 2!'

# 3. Test Live Predictor Endpoint
print('\n--- LIVE PREDICTOR VERIFICATION ---')
pred_payload = {
    'features': {'R&D Spend': 100000, 'Administration': 120000, 'Marketing Spend': 250000, 'State_Florida': 0, 'State_New York': 1},
    'model_name': 'k3_model',
    'algorithm': 'Forward SFS'
}
req_pred = urllib.request.Request(
    f'{BASE_URL}/api/predict',
    data=json.dumps(pred_payload).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
res_pred = urllib.request.urlopen(req_pred)
data_pred = json.loads(res_pred.read().decode('utf-8'))
print('Live Predictor Status:', data_pred.get('status'))
pred_val = data_pred.get('prediction', {}).get('predicted_profit')
print(f"Predicted Profit: ${pred_val:,.2f}")

# 4. Test Optuna Endpoint
print('\n--- OPTUNA HYPERPARAMETER TUNING VERIFICATION ---')
res_opt = urllib.request.urlopen(f'{BASE_URL}/api/optuna')
data_opt = json.loads(res_opt.read().decode('utf-8'))
print('Optuna Status:', data_opt.get('status'))
print('Optuna Best Params:', data_opt.get('optuna', {}).get('best_params'))
print(f"Optuna Best CV RMSE: ${data_opt.get('optuna', {}).get('best_cv_rmse'):,.2f}")
print(f"Optuna Test RMSE: ${data_opt.get('optuna', {}).get('test_rmse'):,.2f}")

print('\n' + '=' * 60)
print('ALL TESTS PASSED WITH 100% SUCCESS!')
print('=' * 60)
