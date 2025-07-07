# forecasting_project.py

"""
Sales Forecasting Project

Goal: Predict weekly sales (quantity) for Sept–Nov 2024 using historical data.
Steps:
1. Data Loading & Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training (Random Forest)
5. Validation on June–Aug 2024
6. Forecasting Sept–Nov 2024
7. Saving outputs
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# 1. Load Data
df = pd.read_csv('data/Assessment-2-Associate-DS(in).csv')
df['week_end_date'] = pd.to_datetime(df['week_end_date'])

# 2. Sort and create time features
df = df.sort_values('week_end_date')
df['week'] = df['week_end_date'].dt.isocalendar().week
df['month'] = df['week_end_date'].dt.month
df['year'] = df['week_end_date'].dt.year

# 3. Train / Validation Split
train_df = df[df['week_end_date'] < '2024-06-01']
val_df = df[(df['week_end_date'] >= '2024-06-01') & (df['week_end_date'] <= '2024-08-31')]
test_df = df[df['week_end_date'] > '2024-08-31']

# 4. Feature selection
features = ['channel', 'brand', 'category', 'sub_category', 'week', 'month', 'year']
df_encoded = pd.get_dummies(df, columns=['channel', 'brand', 'category', 'sub_category'])
feature_cols = [col for col in df_encoded.columns if col not in ['week_end_date', 'quantity', 'SerialNum']]

# 5. Modeling per SerialNum
serials = df['SerialNum'].unique()
forecasts = []

for serial in serials:
    print(f"Processing SerialNum: {serial}")
    serial_df = df[df['SerialNum'] == serial].copy()
    serial_df = pd.get_dummies(serial_df, columns=['channel', 'brand', 'category', 'sub_category'])
    
    serial_df['week'] = serial_df['week_end_date'].dt.isocalendar().week
    serial_df['month'] = serial_df['week_end_date'].dt.month
    serial_df['year'] = serial_df['week_end_date'].dt.year

    train = serial_df[serial_df['week_end_date'] < '2024-06-01']
    val = serial_df[(serial_df['week_end_date'] >= '2024-06-01') & (serial_df['week_end_date'] <= '2024-08-31')]
    test = serial_df[serial_df['week_end_date'] > '2024-08-31']

    X_train = train[feature_cols]
    y_train = train['quantity']
    X_val = val[feature_cols]
    y_val = val['quantity']
    X_test = test[feature_cols]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Validation
    val_preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_preds)
    print(f"Validation MAE for SerialNum {serial}: {mae:.2f}")

    # Forecasting
    test_preds = model.predict(X_test)
    test['forecast_quantity'] = test_preds
    test['SerialNum'] = serial

    forecasts.append(test[['week_end_date', 'SerialNum', 'forecast_quantity']])

    # Save model
    joblib.dump(model, f'models/model_serial_{serial}.pkl')

# Combine forecast
forecast_df = pd.concat(forecasts)
forecast_df.to_csv('outputs/forecast_per_serialnum_Sep_Nov.csv', index=False)
print("✅ Forecasting complete and saved to outputs/forecast_per_serialnum_Sep_Nov.csv")
