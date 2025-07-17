#!/usr/bin/env python3
"""
Simplified YR Tax Prediction Pipeline

Focuses on training and evaluating the optimized YR prediction model.
Retains essential printing and progress bars.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import glob
import os
from tqdm import tqdm
import gc
import warnings

warnings.filterwarnings('ignore')

def load_and_combine_data():
    """Load and combine all parquet files."""
    print("🔄 Step 1: Loading and combining data...")
    data_files = glob.glob('yqyr_data/*.parquet')
    data_files.sort()
    print(f"Found {len(data_files)} data files")
    
    dfs = [pd.read_parquet(file) for file in tqdm(data_files, desc="Loading files")]
    combined_df = pd.concat(dfs, ignore_index=True)
    del dfs; gc.collect()
    
    print(f"✅ Combined dataset: {combined_df.shape}")
    return combined_df

def preprocess_data(df):
    """Preprocess and engineer features."""
    print("\n🔄 Step 2: Preprocessing and feature engineering...")
    
    df['yq'] = df['tax_yq_amount'].fillna(0)
    df['yr'] = df['tax_yr_amount'].fillna(0)
    
    print(f"YQ stats: mean=${df['yq'].mean():.2f}, non-zero={df['yq'].gt(0).mean()*100:.1f}%")
    print(f"YR stats: mean=${df['yr'].mean():.2f}, non-zero={df['yr'].gt(0).mean()*100:.1f}%")
    
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['hour'] = df['timestamp_dt'].dt.hour
    df['day_of_week'] = df['timestamp_dt'].dt.dayofweek
    df['month'] = df['timestamp_dt'].dt.month
    
    df['total_duration'] = df['inboundDuration'] + df['outboundDuration']
    df['is_round_trip'] = (df['returnDate'] > 0).astype(int)
    df['same_carrier'] = (df['inboundOperatingCarrier'] == df['outboundOperatingCarrier']).astype(int)
    
    df['price_tier'] = pd.cut(df['totalAmount'], 
                             bins=[0, 500, 2000, float('inf')], 
                             labels=[0, 1, 2]).astype(int)
    df['early_booking'] = (df['advancePurchase'] >= 30).astype(int)
    
    categorical_cols = [
        'originCityCode', 'destinationCityCode', 'currencyCode',
        'validatingCarrier', 'inboundOperatingCarrier', 'outboundOperatingCarrier'
    ]
    
    for col in tqdm(categorical_cols, desc="Encoding categories"):
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    
    feature_cols = [
        'totalAmount', 'advancePurchase', 'lengthOfStay',
        'inboundDuration', 'outboundDuration', 'total_duration',
        'validatingCarrier_encoded', 'inboundOperatingCarrier_encoded', 
        'outboundOperatingCarrier_encoded', 'price_tier', 'early_booking'
    ]
    
    X = df[feature_cols].copy()
    targets = {'yq': df['yq'].values, 'yr': df['yr'].values}
    
    print(f"✅ Feature matrix: {X.shape}")
    return X, targets, feature_cols

def calculate_hit_rate(y_true, y_pred, tolerance=2.0):
    """Calculate hit rate - predictions within tolerance."""
    zero_mask = y_true == 0
    zero_hits = np.abs(y_pred[zero_mask]) <= tolerance
    
    nonzero_mask = y_true > 0
    nonzero_hits = np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask]) <= tolerance
    
    total_hits = zero_hits.sum() + nonzero_hits.sum()
    
    return {
        'overall_hit_rate': total_hits / len(y_true),
        'zero_hit_rate': zero_hits.mean() if zero_mask.sum() > 0 else 0,
        'nonzero_hit_rate': nonzero_hits.mean() if nonzero_mask.sum() > 0 else 0
    }

def train_model(X, y, params, num_boost_round, early_stopping_rounds, desc):
    """Train a LightGBM model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    with tqdm(total=num_boost_round, desc=desc) as pbar:
        model = lgb.train(
            params, train_data, valid_sets=[valid_data],
            num_boost_round=num_boost_round,
            callbacks=[lambda env: pbar.update(1), lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)]
        )
    
    y_pred = model.predict(X_test)
    
    results = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'hit_rate': calculate_hit_rate(y_test, y_pred),
        'y_test': y_test, 
        'y_pred': y_pred  
    }
    
    return model, results

def report_and_save_results(model, results, feature_cols):
    """Print evaluation metrics, feature importance, and persist artifacts."""
    print("\n📊 Step 3: Results\n" + "=" * 60)

    # Core metrics
    print(f"RMSE: {results['rmse']:.2f} | MAE: {results['mae']:.2f} | R²: {results['r2']:.4f}")

    # Hit rates for multiple tolerances
    tolerances = [1, 2, 5]
    hit_rates = {t: calculate_hit_rate(results['y_test'], results['y_pred'], tolerance=float(t)) for t in tolerances}
    for t in tolerances:
        label = "(±$2)" if t == 2 else f"(±${t})"
        print(f"Hit Rate {label}: {hit_rates[t]['overall_hit_rate']*100:.1f}%")

    # Breakdown for ±$2 tolerance
    two_tol = hit_rates[2]
    print(f"  • Zero: {two_tol['zero_hit_rate']*100:.1f}% | Non-zero: {two_tol['nonzero_hit_rate']*100:.1f}%")

    # Feature importance (top-5)
    importance = model.feature_importance(importance_type='gain')
    feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importance}).nlargest(5, 'importance')
    print("\nTop 5 Features:")
    for i, row in enumerate(feat_imp.itertuples(index=False), 1):
        print(f"  {i}. {row.feature:<25} {row.importance:>10.0f}")

    # Persist artifacts
    print("\n💾 Saving artifacts…")
    model.save_model('yr_final_model.txt')
    pd.Series(feature_cols).to_csv('feature_names.csv', index=False, header=['feature'])
    with open('model_results.txt', 'w') as f:
        f.write("YR Tax Prediction Model Results\n" + "=" * 40 + "\n")
        f.write(f"RMSE: {results['rmse']:.2f}\nMAE: {results['mae']:.2f}\nR²: {results['r2']:.4f}\n")
        for t in tolerances:
            f.write(f"Hit Rate (±${t}): {hit_rates[t]['overall_hit_rate']*100:.1f}%\n")
        f.write(f"Features: {len(feature_cols)}\n")
    print("✅ Artifacts saved: yr_final_model.txt, feature_names.csv, model_results.txt")

def main():
    """Main pipeline execution."""
    print("🚀 YR TAX PREDICTION PIPELINE")
    print("=" * 50)
    
    df = load_and_combine_data()
    X, targets, feature_cols = preprocess_data(df)
    del df; gc.collect()
    
    # Optimized model training
    optimized_params = {'objective': 'regression', 'metric': 'rmse', 'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'verbose': -1, 'random_state': 42}
    optimized_model, optimized_results = train_model(X, targets['yr'], optimized_params, 100, 15, "Optimized training")
    
    report_and_save_results(optimized_model, optimized_results, feature_cols)
    
    print("\n🎉 PIPELINE COMPLETE!")
    print(f"🎯 Final Result: {optimized_results['hit_rate']['overall_hit_rate']*100:.1f}% within $2")

if __name__ == "__main__":
    main() 