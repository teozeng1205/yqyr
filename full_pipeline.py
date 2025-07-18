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

def get_yq_banned_airports():
    """Return set of airport codes where YQ taxes are typically banned/restricted."""
    # Common airports/countries where YQ taxes are banned or restricted
    banned_codes = {
        # Brazil airports (BR country code in data)
        'GRU', 'CGH', 'BSB', 'RIO', 'SDU', 'GIG', 'SSA', 'FOR', 'REC', 'MAO',
        'BEL', 'CWB', 'FLN', 'POA', 'VIX', 'MCZ', 'NAT', 'AJU', 'JPA', 'THE',
        # Hong Kong 
        'HKG',
        # Philippines airports (PH country code)
        'MNL', 'CEB', 'DVO', 'ILO', 'CRK', 'KLO', 'TAG', 'BCD', 'CGY', 'DGT',
        # Additional commonly restricted airports
        'TPE', 'KHH',  # Taiwan
        'ICN', 'GMP',  # South Korea
        'NRT', 'HND',  # Japan (some restrictions)
    }
    return banned_codes

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

def preprocess_data(df, use_cache=True):
    """Preprocess and engineer features."""
    print("\n🔄 Step 2: Preprocessing and feature engineering...")
    
    # Check for cached preprocessed data
    cache_file = 'preprocessed_data.pkl'
    if use_cache and os.path.exists(cache_file):
        print("📁 Loading cached preprocessed data...")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                print("✅ Loaded cached preprocessed data")
                return cached_data
        except Exception as e:
            print(f"⚠️ Error loading cache: {e}. Proceeding with fresh preprocessing...")
    
    df['yq'] = df['tax_yq_amount'].fillna(0)
    df['yr'] = df['tax_yr_amount'].fillna(0)
    
    print(f"YQ stats: mean=${df['yq'].mean():.2f}, non-zero={df['yq'].gt(0).mean()*100:.1f}%")
    print(f"YR stats: mean=${df['yr'].mean():.2f}, non-zero={df['yr'].gt(0).mean()*100:.1f}%")
    
    # Remove leakage: create base fare proxy instead of using raw totalAmount
    print("Creating base fare proxy to remove leakage...")
    est_other_taxes = 50  # Estimate for other taxes/fees not captured in yq/yr
    df['base_fare'] = df['totalAmount'] - df['yq'] - df['yr'] - est_other_taxes
    df['base_fare'] = df['base_fare'].clip(lower=0)  # Ensure non-negative
    
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['hour'] = df['timestamp_dt'].dt.hour
    df['day_of_week'] = df['timestamp_dt'].dt.dayofweek
    df['month'] = df['timestamp_dt'].dt.month
    
    # Domain Feature 1: Great-circle distance between origin & destination
    print("Adding great-circle distance feature...")
    city_locations = load_city_locations()
    if city_locations is not None:
        # Merge origin coordinates
        df = df.merge(
            city_locations[['latitude', 'longitude']].rename(columns={
                'latitude': 'origin_lat', 'longitude': 'origin_lon'
            }),
            left_on='originCityCode', right_index=True, how='left'
        )
        # Merge destination coordinates
        df = df.merge(
            city_locations[['latitude', 'longitude']].rename(columns={
                'latitude': 'dest_lat', 'longitude': 'dest_lon'
            }),
            left_on='destinationCityCode', right_index=True, how='left'
        )
        
        # Calculate distance for valid coordinates
        valid_coords = df[['origin_lat', 'origin_lon', 'dest_lat', 'dest_lon']].notna().all(axis=1)
        df['distance_km'] = 0.0
        if valid_coords.sum() > 0:
            df.loc[valid_coords, 'distance_km'] = haversine_distance(
                df.loc[valid_coords, 'origin_lat'],
                df.loc[valid_coords, 'origin_lon'],
                df.loc[valid_coords, 'dest_lat'],
                df.loc[valid_coords, 'dest_lon']
            )
        
        print(f"✅ Added distance for {valid_coords.sum()}/{len(df)} routes")
        # Clean up temporary columns
        df.drop(['origin_lat', 'origin_lon', 'dest_lat', 'dest_lon'], axis=1, inplace=True)
    else:
        df['distance_km'] = 0.0
        print("⚠️ Distance feature set to 0 (no location data)")
    
    # Domain Feature 2: Days to departure and bucketing
    print("Adding days-to-departure feature...")
    if 'outboundDate' in df.columns:
        # Convert outbound date from timestamp to date
        df['outbound_dt'] = pd.to_datetime(df['outboundDate'], unit='ms', errors='coerce')
        # Calculate days to departure
        df['days_to_departure'] = (df['outbound_dt'] - df['timestamp_dt']).dt.days
        
        # Bucket days to departure (0-7, 8-21, >21)
        df['departure_bucket'] = pd.cut(
            df['days_to_departure'],
            bins=[-float('inf'), 7, 21, float('inf')],
            labels=[0, 1, 2]  # 0-7 days, 8-21 days, >21 days
        ).astype(int)
        
        print(f"✅ Days to departure: mean={df['days_to_departure'].mean():.1f}")
    else:
        # Fallback to advance purchase if outboundDate not available
        df['days_to_departure'] = df['advancePurchase']
        df['departure_bucket'] = pd.cut(
            df['days_to_departure'],
            bins=[-float('inf'), 7, 21, float('inf')],
            labels=[0, 1, 2]
        ).astype(int)
        print("⚠️ Using advancePurchase as days_to_departure proxy")
    
    # Domain Feature 3: YQ banned airports flag
    print("Adding YQ banned airports flag...")
    banned_airports = get_yq_banned_airports()
    df['is_origin_yq_banned'] = df['originCityCode'].isin(banned_airports).astype(int)
    df['is_dest_yq_banned'] = df['destinationCityCode'].isin(banned_airports).astype(int)
    df['is_route_yq_banned'] = ((df['is_origin_yq_banned'] == 1) | 
                                (df['is_dest_yq_banned'] == 1)).astype(int)
    
    banned_origin_count = df['is_origin_yq_banned'].sum()
    banned_dest_count = df['is_dest_yq_banned'].sum()
    banned_route_count = df['is_route_yq_banned'].sum()
    print(f"✅ YQ banned: {banned_origin_count} origins, {banned_dest_count} destinations, {banned_route_count} routes")
    
    df['total_duration'] = df['inboundDuration'] + df['outboundDuration']
    df['is_round_trip'] = (df['returnDate'] > 0).astype(int)
    df['same_carrier'] = (df['inboundOperatingCarrier'] == df['outboundOperatingCarrier']).astype(int)
    
    df['price_tier'] = pd.cut(df['base_fare'], 
                             bins=[0, 400, 1500, float('inf')], 
                             labels=[0, 1, 2]).astype(int)
    df['early_booking'] = (df['advancePurchase'] >= 30).astype(int)
    
    categorical_cols = [
        'originCityCode', 'destinationCityCode', 'currencyCode',
        'validatingCarrier', 'inboundOperatingCarrier', 'outboundOperatingCarrier'
    ]
    
    for col in tqdm(categorical_cols, desc="Encoding categories"):
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    
    # Target encoding for wide keys
    print("Creating target-encoded features...")
    df['route_te'] = target_encode_features(
        df, 'yq', ['originCityCode', 'destinationCityCode']
    )
    df['carrier_route_te'] = target_encode_features(
        df, 'yq', ['validatingCarrier', 'originCityCode', 'destinationCityCode']
    )
    
    # Identify categorical features for LightGBM
    categorical_features = [
        'originCityCode_encoded', 'destinationCityCode_encoded', 'currencyCode_encoded',
        'validatingCarrier_encoded', 'inboundOperatingCarrier_encoded', 
        'outboundOperatingCarrier_encoded', 'price_tier', 'departure_bucket',
        'is_origin_yq_banned', 'is_dest_yq_banned', 'is_route_yq_banned'
    ]
    
    feature_cols = [
        'base_fare', 'advancePurchase', 'lengthOfStay', 'days_to_departure',
        'inboundDuration', 'outboundDuration', 'total_duration', 'distance_km',
        'validatingCarrier_encoded', 'inboundOperatingCarrier_encoded', 
        'outboundOperatingCarrier_encoded', 'price_tier', 'early_booking',
        'departure_bucket', 'is_origin_yq_banned', 'is_dest_yq_banned', 'is_route_yq_banned',
        'route_te', 'carrier_route_te', 'hour', 'day_of_week', 'month'
    ]
    
    X = df[feature_cols].copy()
    targets = {'yq': df['yq'].values, 'yr': df['yr'].values}
    
    # Cache the preprocessed data
    result = (X, targets, feature_cols, categorical_features)
    if use_cache:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
                print("💾 Saved preprocessed data to cache")
        except Exception as e:
            print(f"⚠️ Could not save cache: {e}")
    
    print(f"✅ Feature matrix: {X.shape}")
    print(f"✅ Categorical features: {len(categorical_features)}")
    print(f"✅ Domain features added: distance_km, days_to_departure, departure_bucket, YQ_banned flags")
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
    
    # Create grouping key
    group_key = df[key_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    
    for train_idx, val_idx in kfold.split(df):
        # Calculate means on training fold
        train_df = df.iloc[train_idx]
        train_group_key = group_key.iloc[train_idx]
        
        group_means = train_df.groupby(train_group_key)[target_col].mean()
        global_mean = train_df[target_col].mean()
        
        # Apply to validation fold
        val_group_key = group_key.iloc[val_idx]
        encoded_values[val_idx] = val_group_key.map(group_means).fillna(global_mean)
    
    return pd.Series(encoded_values, index=df.index)

def create_validation_split(df, test_size=0.2, method='carrier_route', random_state=42):
    """
    Create better validation split based on carrier-route combinations or date blocks.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe
    test_size : float
        Proportion of data for test set
    method : str
        Split method: 'carrier_route' or 'date'
    random_state : int
        Random state for reproducibility
        
    Returns
    -------
    train_idx, test_idx : arrays
        Indices for train and test sets
    """
    np.random.seed(random_state)
    
    if method == 'carrier_route':
        # Group by (validatingCarrier, originCityCode, destinationCityCode)
        group_key = df[['validatingCarrier', 'originCityCode', 'destinationCityCode']].apply(
            lambda x: '_'.join(x.astype(str)), axis=1
        )
        unique_groups = group_key.unique()
        
        # Randomly select groups for test set
        n_test_groups = max(1, int(len(unique_groups) * test_size))
        test_groups = np.random.choice(unique_groups, size=n_test_groups, replace=False)
        
        test_mask = group_key.isin(test_groups)
        
    elif method == 'date':
        # Sort by timestamp and split by date blocks
        df_sorted = df.sort_values('timestamp')
        split_idx = int(len(df_sorted) * (1 - test_size))
        
        test_mask = pd.Series(False, index=df.index)
        test_mask.loc[df_sorted.index[split_idx:]] = True
        
    else:
        raise ValueError("Method must be 'carrier_route' or 'date'")
    
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
    train_idx, test_idx = create_validation_split(df_full, test_size=0.2, method='carrier_route')
    
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

def optimize_hyperparameters(X, y, df_full, categorical_features, n_trials=15):
    """Use Optuna to tune LightGBM hyperparameters to maximize hit rate (±10%)."""
    import optuna  # type: ignore  # Imported lazily to avoid linter resolution issues

    def objective(trial):
        # Define the search space
        param_grid = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            'boosting_type': 'gbdt',
            'random_state': 42,
            'num_leaves': trial.suggest_int('num_leaves', 31, 255, step=16),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
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
        pct_hit_rate = calculate_percentage_hit_rate(results['y_test'], results['y_pred'], percent_tolerance=0.05)['overall_hit_rate']
        return 1.0 - pct_hit_rate

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n🏆 Best hit rate: {(1 - study.best_value) * 100:.2f}% within 5%")

    # Merge constant parameters with the best hyperparameters found
    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'verbose': -1,
        'random_state': 42,
    })

    return best_params

def report_and_save_results(model, results, feature_cols, df_all):
    """Print evaluation metrics, feature importance, grouped by airline."""
    print("\n📊 Step 3: Results\n" + "=" * 60)

    # Core metrics
    print(f"RMSE: {results['rmse']:.2f} | MAE: {results['mae']:.2f} | R²: {results['r2']:.4f}")

    # ---------------- Absolute $ tolerance hit rates ----------------
    tolerances = [1, 2, 5]
    hit_rates = {t: calculate_hit_rate(results['y_test'], results['y_pred'], tolerance=float(t)) for t in tolerances}
    for t in tolerances:
        label = "(±$2)" if t == 2 else f"(±${t})"
        print(f"Hit Rate {label}: {hit_rates[t]['overall_hit_rate']*100:.1f}%")

    # Breakdown for ±$2 tolerance
    two_tol = hit_rates[2]
    print(f"  • Zero: {two_tol['zero_hit_rate']*100:.1f}% | Non-zero: {two_tol['nonzero_hit_rate']*100:.1f}%")

    # ---------------- Percentage tolerance hit rates ----------------
    pct_tols = [0.01, 0.02, 0.05]
    pct_hit_rates = {p: calculate_percentage_hit_rate(results['y_test'], results['y_pred'], percent_tolerance=p) for p in pct_tols}

    for p in pct_tols:
        pct_label = int(p * 100)
        print(f"Hit Rate (±{pct_label}%): {pct_hit_rates[p]['overall_hit_rate']*100:.1f}%")

    # ---------------- Hit rates by validating airline (Detailed) ----------------
    print("\n✈️  Hit Rates by Validating Airline (Detailed)")
    test_idx = results['test_index']
    airline_series = df_all.loc[test_idx, 'validatingCarrier']

    # Get unique carriers and sort them by overall hit rate (e.g., ±$2)
    carrier_performance = {}
    for carrier in airline_series.unique():
        carrier_mask = airline_series == carrier
        y_test_carrier = results['y_test'][carrier_mask]
        y_pred_carrier = results['y_pred'][carrier_mask]
        
        # Calculate overall 2$ hit rate for sorting
        hr_2_abs = calculate_hit_rate(y_test_carrier, y_pred_carrier, tolerance=2.0)['overall_hit_rate']
        carrier_performance[carrier] = hr_2_abs
    
    sorted_carriers = sorted(carrier_performance.items(), key=lambda x: x[1], reverse=True)

    # Define tolerances here so they are accessible
    tolerances = [1, 2, 5]
    pct_tols = [0.01, 0.02, 0.05]

    for carrier, _ in sorted_carriers:
        carrier_mask = airline_series == carrier
        y_test_carrier = results['y_test'][carrier_mask]
        y_pred_carrier = results['y_pred'][carrier_mask]

        print(f"\nCarrier: {carrier}")
        
        # Absolute $ tolerances
        print("  Absolute Tolerances:")
        for t in tolerances:
            hr = calculate_hit_rate(y_test_carrier, y_pred_carrier, tolerance=float(t))['overall_hit_rate']
            print(f"    ±${t}: {hr*100:.1f}%")

        # Percentage tolerances
        print("  Percentage Tolerances:")
        for p in pct_tols:
            hr = calculate_percentage_hit_rate(y_test_carrier, y_pred_carrier, percent_tolerance=p)['overall_hit_rate']
            pct_label = int(p * 100)
            print(f"    ±{pct_label}%: {hr*100:.1f}%")
        print("-" * 60) # Separator for readability

    # Feature importance (top-5)
    importance = model.feature_importance(importance_type='gain')
    feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importance}).nlargest(5, 'importance')
    print("\nTop 5 Features:")
    for i, row in enumerate(feat_imp.itertuples(index=False), 1):
        print(f"  {i}. {row.feature:<25} {row.importance:>10.0f}")

    # Persist artifacts
    print("\n💾 Saving artifacts…")
    model.save_model('yq_final_model.txt')
    pd.Series(feature_cols).to_csv('feature_names.csv', index=False, header=['feature'])
    with open('model_results.txt', 'w') as f:
        f.write("YQ Tax Prediction Model Results\n" + "=" * 40 + "\n")
        f.write(f"RMSE: {results['rmse']:.2f}\nMAE: {results['mae']:.2f}\nR²: {results['r2']:.4f}\n")
        for t in tolerances:
            f.write(f"Hit Rate (±${t}): {hit_rates[t]['overall_hit_rate']*100:.1f}%\n")
        f.write(f"Features: {len(feature_cols)}\n")
    print("✅ Artifacts saved: yq_final_model.txt, feature_names.csv, model_results.txt")

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
    final_results['pct_hit_rate'] = calculate_percentage_hit_rate(final_results['y_test'], final_results['y_pred'], percent_tolerance=0.05)

    report_and_save_results(final_model, final_results, feature_cols, df)

    # Now safe to release df
    del df; gc.collect()
    
    print("\n🎉 PIPELINE COMPLETE!")
    print(f"🎯 Final Result: {final_results['pct_hit_rate']['overall_hit_rate']*100:.1f}% within 5%")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YQ Tax Prediction Pipeline")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    parser.add_argument("--no-cache", action="store_true", help="Disable preprocessing cache")
    args = parser.parse_args()

    main(tune_hyperparameters=args.tune, use_cache=not args.no_cache) 