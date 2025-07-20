#!/usr/bin/env python3
"""
Simplified YR Tax Prediction Pipeline

Focuses on training and evaluating the optimized YR prediction model.
Retains essential printing and progress bars.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import glob
import os
from tqdm import tqdm
import gc
import warnings
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
# tqdm for progress bars (already imported), optuna will be imported lazily within the tuning function

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Domain feature helper functions

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on earth (in kilometers).
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    return c * r

def load_city_locations():
    """Load city location data for distance calculations."""
    try:
        city_locations = pd.read_csv('citylocation.csv', sep='\t')
        print(f"✅ Loaded {len(city_locations)} city locations")
        return city_locations.set_index('citycode')
    except Exception as e:
        print(f"⚠️ Could not load citylocation.csv: {e}")
        return None

# ---------------------------------------------------------------------------
# Helper metric: percentage-based hit rate

def calculate_percentage_hit_rate(y_true: np.ndarray, y_pred: np.ndarray, percent_tolerance: float = 0.02):
    """Return hit rate where prediction is within a percentage of the true value.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth targets.
    y_pred : array-like of shape (n_samples,)
        Predicted targets.
    percent_tolerance : float, default 0.02 (i.e. 2%)
        Allowed % deviation expressed as a decimal fraction.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    nonzero_mask = y_true > 0
    # For non-zero targets, use relative tolerance
    hits_nonzero = np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask]) <= percent_tolerance * y_true[nonzero_mask]

    # For zero targets, fall back to absolute tolerance based on 1 unit of value times %
    zero_mask = ~nonzero_mask
    abs_tol_for_zero = percent_tolerance  # Treat as absolute value for zero target
    hits_zero = np.abs(y_pred[zero_mask]) <= abs_tol_for_zero

    total_hits = hits_nonzero.sum() + hits_zero.sum()
    return {
        'overall_hit_rate': total_hits / len(y_true),
        'zero_hit_rate': hits_zero.mean() if zero_mask.sum() > 0 else 0,
        'nonzero_hit_rate': hits_nonzero.mean() if nonzero_mask.sum() > 0 else 0
    }

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

def _add_time_features(df):
    """Adds time-based features like departure day/month and buckets."""
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['hour'] = df['timestamp_dt'].dt.hour
    df['day_of_week'] = df['timestamp_dt'].dt.dayofweek
    df['month'] = df['timestamp_dt'].dt.month

    # Days to departure
    if 'outboundDate' in df.columns:
        df['outbound_dt'] = pd.to_datetime(df['outboundDate'], unit='ms', errors='coerce')
        df['days_to_departure'] = (df['outbound_dt'] - df['timestamp_dt']).dt.days
    else:
        df['days_to_departure'] = df['advancePurchase']  # Fallback
    
    df['departure_bucket'] = pd.cut(df['days_to_departure'], bins=[-float('inf'), 7, 21, float('inf')], labels=[0, 1, 2]).astype(int)
    return df

def _add_domain_features(df):
    """Adds domain-specific features like distance and other trip attributes."""
    # Great-circle distance
    city_locations = load_city_locations()
    if city_locations is not None:
        df = df.merge(city_locations.add_prefix('origin_'), left_on='originCityCode', right_index=True, how='left')
        df = df.merge(city_locations.add_prefix('dest_'), left_on='destinationCityCode', right_index=True, how='left')
        
        valid_coords = df[['origin_latitude', 'origin_longitude', 'dest_latitude', 'dest_longitude']].notna().all(axis=1)
        df['distance_km'] = 0.0
        if valid_coords.any():
            df.loc[valid_coords, 'distance_km'] = haversine_distance(
                df.loc[valid_coords, 'origin_latitude'], df.loc[valid_coords, 'origin_longitude'],
                df.loc[valid_coords, 'dest_latitude'], df.loc[valid_coords, 'dest_longitude']
            )
        df.drop(columns=[col for col in df.columns if 'latitude' in col or 'longitude' in col], inplace=True)
    else:
        df['distance_km'] = 0.0
    
    df['total_duration'] = df['inboundDuration'] + df['outboundDuration']
    df['is_round_trip'] = (df['returnDate'] > 0).astype(int)
    df['same_carrier'] = (df['inboundOperatingCarrier'] == df['outboundOperatingCarrier']).astype(int)
    return df

def _add_engineered_features(df):
    """Adds ratio, bucketed, and log-transformed features."""
    df['price_tier'] = pd.cut(df['base_fare'], bins=[0, 400, 1500, float('inf')], labels=[0, 1, 2]).astype(int)
    df['early_booking'] = (df['advancePurchase'] >= 30).astype(int)
    df['price_per_km'] = df['base_fare'] / (df['distance_km'] + 1)
    df['price_per_day'] = df['base_fare'] / (df['lengthOfStay'] + 1)
    df['distance_bucket'] = pd.cut(df['distance_km'], bins=[-float('inf'), 1000, 3000, float('inf')], labels=[0, 1, 2]).astype(int)
    df['log_base_fare'] = np.log1p(df['base_fare'])
    df['log_distance_km'] = np.log1p(df['distance_km'])
    df['log_price_per_km'] = np.log1p(df['price_per_km'])
    df['adv_purchase_bucket'] = pd.cut(df['advancePurchase'], bins=[-float('inf'), 3, 7, 14, 30, float('inf')], labels=[0, 1, 2, 3, 4]).astype(int)
    df['season'] = df['month'].apply(lambda m: (m - 1) // 3)
    return df

def _encode_categorical_features(df):
    """Encodes categorical columns using LabelEncoder and target encoding."""
    # Label Encoding
    categorical_cols = ['originCityCode', 'destinationCityCode', 'currencyCode', 'validatingCarrier', 'inboundOperatingCarrier', 'outboundOperatingCarrier']
    for col in tqdm(categorical_cols, desc="Encoding categories"):
        df[col + '_encoded'] = LabelEncoder().fit_transform(df[col].astype(str))

    # Target Encoding
    with tqdm(total=3, desc="Overall Target Encoding") as pbar:
        for keys in [['originCityCode', 'destinationCityCode'], ['validatingCarrier', 'originCityCode', 'destinationCityCode'], ['validatingCarrier']]:
            col_name = '_'.join(keys) + '_te'
            df[col_name] = target_encode_features(df, 'yq', keys)
            pbar.update(1)
    
    # Manually rename for consistency with original feature list
    df.rename(columns={
        'originCityCode_destinationCityCode_te': 'route_te',
        'validatingCarrier_originCityCode_destinationCityCode_te': 'carrier_route_te',
        'validatingCarrier_te': 'carrier_te'
    }, inplace=True)
    return df

def preprocess_data(df, use_cache=True):
    """Preprocess and engineer features."""
    print("\n🔄 Step 2: Preprocessing and feature engineering...")
    
    cache_file = 'preprocessed_data.pkl'
    if use_cache and os.path.exists(cache_file):
        print("📁 Loading cached preprocessed data...")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Error loading cache: {e}. Proceeding with fresh preprocessing...")
    
    df['yq'] = df['tax_yq_amount'].fillna(0)
    df['yr'] = df['tax_yr_amount'].fillna(0)
    df['base_fare'] = df['totalAmount']
    print(f"YQ stats: mean=${df['yq'].mean():.2f}, non-zero={df['yq'].gt(0).mean()*100:.1f}%")
    
    # --- Feature Engineering Pipeline ---
    df = _add_time_features(df)
    df = _add_domain_features(df)
    df = _add_engineered_features(df)
    df = _encode_categorical_features(df)
    
    # --- Final Feature Selection ---
    categorical_features = [
        'originCityCode_encoded', 'destinationCityCode_encoded', 'currencyCode_encoded',
        'validatingCarrier_encoded', 'inboundOperatingCarrier_encoded', 
        'outboundOperatingCarrier_encoded', 'price_tier', 'departure_bucket',
        'distance_bucket', 'adv_purchase_bucket', 'season'
    ]
    
    feature_cols = [
        'base_fare', 'advancePurchase', 'lengthOfStay', 'days_to_departure',
        'inboundDuration', 'outboundDuration', 'total_duration', 'distance_km',
        'price_per_km', 'price_per_day', 'log_base_fare', 'log_distance_km', 'log_price_per_km',
        'validatingCarrier_encoded', 'inboundOperatingCarrier_encoded', 'outboundOperatingCarrier_encoded',
        'price_tier', 'early_booking', 'departure_bucket', 'distance_bucket', 'adv_purchase_bucket', 'season',
        'route_te', 'carrier_route_te', 'carrier_te',
        'hour', 'day_of_week', 'month'
    ]
    
    X = df[feature_cols].copy()
    targets = {'yq': df['yq'].values, 'yr': df['yr'].values}
    
    result = (X, targets, feature_cols, categorical_features)
    if use_cache:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            print("💾 Saved preprocessed data to cache")
        except Exception as e:
            print(f"⚠️ Could not save cache: {e}")
    
    print(f"✅ Feature matrix: {X.shape}")
    return result

def target_encode_features(df, target_col, key_cols, n_splits=5, random_state=42):
    """
    Perform 5-fold out-of-fold target encoding for specified key columns.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe
    target_col : str
        Name of target column to encode against
    key_cols : list
        List of column names to group by for encoding
    n_splits : int
        Number of folds for cross-validation
    random_state : int
        Random state for reproducibility
        
    Returns
    -------
    encoded_values : Series
        Target-encoded values aligned with input dataframe
    """
    encoded_values = np.zeros(len(df))
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Create grouping key once outside the loop
    group_key_series = df[key_cols].astype(str).agg('_'.join, axis=1)

    for fold, (train_idx, val_idx) in tqdm(enumerate(kfold.split(df)), total=n_splits, desc="Target Encoding Folds"):
        # Calculate means on training fold
        train_df = df.iloc[train_idx]
        train_group_key = group_key_series.iloc[train_idx]
        
        group_means = train_df.groupby(train_group_key)[target_col].mean()
        global_mean = train_df[target_col].mean()
        
        # Apply to validation fold
        val_group_key = group_key_series.iloc[val_idx]
        encoded_values[val_idx] = val_group_key.map(group_means).fillna(global_mean)
    
    return pd.Series(encoded_values, index=df.index)

def create_validation_split(df, test_size=0.2, method='carrier_route', random_state=42):
    """
    Create validation split using one of three methods:
    1. 'carrier_route'  – group-wise split on (carrier, origin, destination)
    2. 'date'           – chronological split based on timestamp
    3. 'stratified'     – row-level split stratified on (carrier, origin, destination)
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe
    test_size : float
        Proportion of data for test set
    method : str
        Split method: 'carrier_route', 'date', or 'stratified'
    random_state : int
        Random state for reproducibility
        
    Returns
    -------
    train_idx, test_idx : arrays
        Indices for train and test sets
    """
    np.random.seed(random_state)
    
    if method == 'carrier_route':
        # Vectorized factorization of the (validatingCarrier, originCityCode, destinationCityCode) combination
        # This avoids the extremely slow row-wise apply used previously and scales well to tens of millions of rows.
        combo_index = pd.MultiIndex.from_frame(
            df[['validatingCarrier', 'originCityCode', 'destinationCityCode']]
        )
        group_codes, uniques = pd.factorize(combo_index)
        n_unique = len(uniques)

        # Randomly select groups for test set
        n_test_groups = max(1, int(n_unique * test_size))
        test_group_codes = np.random.choice(n_unique, size=n_test_groups, replace=False)

        test_mask = np.isin(group_codes, test_group_codes)
        
    elif method == 'stratified':
        # Stratified 80/20 split on (carrier, origin, destination) combination.
        stratify_key = df[['validatingCarrier', 'originCityCode', 'destinationCityCode']]
        stratify_key = stratify_key.astype(str).agg('_'.join, axis=1)

        try:
            train_idx, test_idx = train_test_split(
                df.index.values,
                test_size=test_size,
                stratify=stratify_key,
                random_state=random_state
            )
        except ValueError:
            # Fallback if some classes are too small for stratification
            train_idx, test_idx = train_test_split(
                df.index.values,
                test_size=test_size,
                random_state=random_state
            )

        return train_idx, test_idx

    elif method == 'date':
        # Sort by timestamp and split by date blocks
        df_sorted = df.sort_values('timestamp')
        split_idx = int(len(df_sorted) * (1 - test_size))
        
        test_mask = pd.Series(False, index=df.index)
        test_mask.loc[df_sorted.index[split_idx:]] = True
        
    else:
        raise ValueError("Method must be 'carrier_route', 'date', or 'stratified'")
    
    train_idx = df.index[~test_mask].values
    test_idx = df.index[test_mask].values

    return train_idx, test_idx

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

def train_model(X, y, params, num_boost_round, early_stopping_rounds, desc, df_full, categorical_features):
    """Train a LightGBM model with better validation split and categorical features."""
    # Use better validation split
    train_idx, test_idx = create_validation_split(df_full, test_size=0.2, method='stratified')
    
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    # Map categorical feature names to indices
    categorical_indices = [X.columns.get_loc(col) for col in categorical_features if col in X.columns]
    
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_indices)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data, categorical_feature=categorical_indices)
    
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
        'pct_hit_rate': None,  # placeholder; filled later
        'test_index': X_test.index,  # save indices for downstream grouping
        'y_test': y_test, 
        'y_pred': y_pred  
    }
    
    return model, results

def optimize_hyperparameters(X, y, df_full, categorical_features, n_trials=25):
    """Use Optuna to tune LightGBM hyperparameters to maximize hit rate (±10%)."""
    import optuna  # type: ignore  # Imported lazily to avoid linter resolution issues

    def objective(trial):
        # Define the search space (tuned for very large datasets to speed up search)
        param_grid = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            'boosting_type': 'gbdt',
            'random_state': 42,

            # --- Tuned ranges ---
            # Larger but coarser range for tree size; step size chosen to limit evaluations
            'num_leaves': trial.suggest_int('num_leaves', 64, 512, step=64),
            # Explicit depth control to avoid overly deep trees that slow training
            'max_depth': trial.suggest_int('max_depth', 6, 16, step=2),

            # Learning rate kept in a tighter, more realistic band for big data
            'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.15, log=True),

            # Column & row sampling (a.k.a. feature_fraction & bagging_fraction)
            'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),

            # Leaf-wise regularisation controls
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100, step=5),

            # Histogram binning – larger bins can improve accuracy but increase memory; tune sparingly
            'max_bin': trial.suggest_int('max_bin', 255, 1023, step=128),

            # L1/L2 regularisation (narrower range speeds up search)
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),
        }

        # Train model with the sampled hyperparameters
        _, results = train_model(
            X,
            y,
            param_grid,
            num_boost_round=300,
            early_stopping_rounds=30,
            desc=f"Optuna trial {trial.number}",
            df_full=df_full,
            categorical_features=categorical_features
        )

        # Optuna minimizes the objective; use (1 - hit_rate) so that higher hit_rate is better
        # For 5% tolerance, use calculate_percentage_hit_rate
        pct_hit_rate = calculate_percentage_hit_rate(results['y_test'], results['y_pred'], percent_tolerance=0.10)['overall_hit_rate']
        return results['rmse']

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n🏆 Best hit rate: {(1 - study.best_value) * 100:.2f}% within 10%")

    # Merge constant parameters with the best hyperparameters found
    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'verbose': -1,
        'random_state': 42,
    })

    return best_params

def _report_hit_rates(y_test, y_pred):
    """Helper to calculate and print absolute and percentage hit rates."""
    print("\n--- Hit Rate Analysis ---")
    # Absolute dollar tolerances
    abs_tolerances = [1, 2, 5, 10]
    for t in abs_tolerances:
        hr = calculate_hit_rate(y_test, y_pred, tolerance=float(t))['overall_hit_rate']
        print(f"Hit Rate (±${t}): {hr*100:.1f}%")

    # Percentage tolerances
    pct_tolerances = [0.01, 0.02, 0.05, 0.10]
    for p in pct_tolerances:
        hr = calculate_percentage_hit_rate(y_test, y_pred, percent_tolerance=p)['overall_hit_rate']
        print(f"Hit Rate (±{int(p*100)}%): {hr*100:.1f}%")

def _report_carrier_performance(results, df_all):
    """Helper to calculate and print performance grouped by carrier."""
    print("\n✈️  Hit Rates by Validating Airline (Sorted by ±$2 Hit Rate)")
    test_idx = results['test_index']
    airline_series = df_all.loc[test_idx, 'validatingCarrier']
    
    # Calculate performance for each carrier
    carrier_performance = {}
    for carrier in airline_series.unique():
        mask = airline_series == carrier
        y_test_carrier, y_pred_carrier = results['y_test'][mask], results['y_pred'][mask]
        if len(y_test_carrier) == 0: continue
        
        carrier_performance[carrier] = {
            'abs_hr_2': calculate_hit_rate(y_test_carrier, y_pred_carrier, tolerance=2.0)['overall_hit_rate'],
            'pct_hr_5': calculate_percentage_hit_rate(y_test_carrier, y_pred_carrier, percent_tolerance=0.05)['overall_hit_rate'],
            'rmse': np.sqrt(mean_squared_error(y_test_carrier, y_pred_carrier)),
            'count': len(y_test_carrier)
        }
        
    # Sort carriers by the primary metric (e.g., ±$2 hit rate)
    sorted_carriers = sorted(carrier_performance.items(), key=lambda item: item[1]['abs_hr_2'], reverse=True)

    # Print results
    for carrier, perf in sorted_carriers:
        print(f"Carrier: {carrier:<4} | Count: {perf['count']:<6} | RMSE: ${perf['rmse']:<5.2f} | HR(±$2): {perf['abs_hr_2']*100:<4.1f}% | HR(±5%): {perf['pct_hr_5']*100:<4.1f}%")

def report_and_save_results(model, results, feature_cols, df_all):
    """Print evaluation metrics, feature importance, and save artifacts."""
    print("\n📊 Step 3: Results\n" + "=" * 60)
    print(f"Overall Metrics: RMSE: {results['rmse']:.2f} | MAE: {results['mae']:.2f} | R²: {results['r2']:.4f}")

    _report_hit_rates(results['y_test'], results['y_pred'])
    _report_carrier_performance(results, df_all)

    # Feature importance (top-5)
    importance = model.feature_importance(importance_type='gain')
    feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importance}).nlargest(5, 'importance')
    print("\nTop 5 Features:")
    for i, row in enumerate(feat_imp.to_dict(orient='records'), 1):
        print(f"  {i}. {row['feature']:<25} {row['importance']:>10.0f}")

    # Persist artifacts
    print("\n💾 Saving artifacts…")
    model.save_model('yq_final_model.txt')
    pd.Series(feature_cols).to_csv('feature_names.csv', index=False, header=['feature'])
    with open('model_results.txt', 'w') as f:
        f.write("YQ Tax Prediction Model Results\n" + "=" * 40 + "\n")
        f.write(f"RMSE: {results['rmse']:.2f}\nMAE: {results['mae']:.2f}\nR²: {results['r2']:.4f}\n")
        # You can add more metrics to the file if needed
    print("✅ Artifacts saved: yq_final_model.txt, feature_names.csv, model_results.txt")

def _create_carrier_subplot(fig, axes, results_df, carriers_to_plot, plot_function, **kwargs):
    """Generic helper to create a grid of plots for specified carriers."""
    import math
    n_carriers = len(carriers_to_plot)
    ncols = 4
    nrows = math.ceil(n_carriers / ncols)
    
    for idx, carrier in enumerate(carriers_to_plot):
        ax = axes[idx // ncols, idx % ncols]
        carrier_data = results_df[results_df['validatingCarrier'] == carrier]
        if carrier_data.empty:
            ax.set_title(f"{carrier} (no data)")
            ax.axis('off')
            continue
        
        plot_function(data=carrier_data, ax=ax, **kwargs)
        ax.set_title(carrier)
        ax.grid(True, linestyle='--', alpha=0.6)
    
    # Hide unused subplots
    for i in range(n_carriers, nrows * ncols):
        fig.delaxes(axes.flatten()[i])

def plot_analysis_for_carriers(y_test, y_pred, df_all, test_idx, carriers_to_plot=['AA', 'B6', 'DL', 'BA', 'VS', 'EI', 'UA']):
    """Generate and save aggregated performance plots for specified carriers."""
    print("\n📈 Generating aggregated performance plots...")
    import math
    os.makedirs('plots', exist_ok=True)

    results_df = pd.DataFrame({
        'y_test': y_test,
        'y_pred': y_pred,
        'validatingCarrier': df_all.loc[test_idx, 'validatingCarrier']
    })
    results_df['abs_error'] = np.abs(results_df['y_test'] - results_df['y_pred'])
    results_df['pct_error'] = np.where(results_df['y_test'] != 0, results_df['abs_error'] / results_df['y_test'], np.nan)

    plot_configs = [
        {'name': 'percentage_error', 'plot_func': sns.histplot, 'args': {'x': 'pct_error', 'bins': 50, 'kde': True}},
        {'name': 'absolute_error', 'plot_func': sns.histplot, 'args': {'x': 'abs_error', 'bins': 50, 'kde': True}},
        {'name': 'pred_vs_true', 'plot_func': sns.scatterplot, 'args': {'x': 'y_test', 'y_pred': 'y_pred', 'alpha': 0.3, 's': 10}}
    ]

    for config in plot_configs:
        n_carriers = len(carriers_to_plot)
        ncols = 4
        nrows = math.ceil(n_carriers / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)
        
        _create_carrier_subplot(fig, axes, results_df, carriers_to_plot, config['plot_func'], **config['args'])
        
        if config['name'] == 'pred_vs_true':
            for ax in axes.flatten():
                lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
                ax.plot(lims, lims, 'r--', lw=2)

        fig.suptitle(f'{config["name"].replace("_", " ").title()} Distribution by Carrier')
        plt.tight_layout(rect=(0, 0, 1, 0.97))
        plt.savefig(f'plots/all_{config["name"]}_distribution.png')
        plt.close(fig)

    print("✅ Aggregated plots saved to 'plots/' directory.")

def main(tune_hyperparameters: bool = False, use_cache: bool = True):
    """Main pipeline execution."""
    print("🚀 YQ TAX PREDICTION PIPELINE")
    print("=" * 50)
    
    df = load_and_combine_data()
    X, targets, feature_cols, categorical_features = preprocess_data(df, use_cache=use_cache)
    # Do not delete df yet – needed for airline grouping metrics
    
    best_params = {}
    if tune_hyperparameters:
        # Hyperparameter tuning
        best_params = optimize_hyperparameters(X, targets['yq'], df, categorical_features, n_trials=50)
    else:
        print("\nSkipping hyperparameter tuning. Using default parameters.")
        best_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            'random_state': 42,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }

    # Train final model using the best hyperparameters
    final_model, final_results = train_model(
        X,
        targets['yq'],
        best_params,
        num_boost_round=300,
        early_stopping_rounds=30,
        desc="Final training",
        df_full=df,
        categorical_features=categorical_features
    )
    
    # Before reporting, update the pct_hit_rate in final_results
    # This is needed because the value was initialized to None and not updated elsewhere.
    # Using the 5% tolerance hit rate as per the final print statement.
    final_results['pct_hit_rate'] = calculate_percentage_hit_rate(final_results['y_test'], final_results['y_pred'], percent_tolerance=0.10)

    report_and_save_results(final_model, final_results, feature_cols, df)

    # Generate and save performance plots
    plot_analysis_for_carriers(final_results['y_test'], final_results['y_pred'], df, final_results['test_index'])

    # Now safe to release df
    del df; gc.collect()
    
    print("\n🎉 PIPELINE COMPLETE!")
    print(f"🎯 Final Result: {final_results['pct_hit_rate']['overall_hit_rate']*100:.1f}% within 10%")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YQ Tax Prediction Pipeline")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    parser.add_argument("--no-cache", action="store_true", help="Disable preprocessing cache")
    args = parser.parse_args()

    main(tune_hyperparameters=args.tune, use_cache=not args.no_cache) 