#!/usr/bin/env python3
"""
Baseline Random Forest Model for YQ/YR Tax Amount Prediction
Treats empty/null values as 0 and converts to classification problem
"""

import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif

# Progress tracking
from tqdm import tqdm
import time

class YQYRPreprocessor:
    """Preprocessor for YQ/YR tax amount data"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.yq_bins = None
        self.yr_bins = None
        
    def create_bins(self, yq_values: np.ndarray, yr_values: np.ndarray, n_bins: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Create bins for classification"""
        # For YQ (less missing values)
        yq_non_zero = yq_values[yq_values > 0]
        self.yq_bins = np.percentile(yq_non_zero, np.linspace(0, 100, n_bins + 1))
        self.yq_bins[0] = 0  # Include zero as first bin
        
        # For YR (more missing values) - use unique values if not many distinct values
        yr_non_zero = yr_values[yr_values > 0]
        if len(yr_non_zero) > 0:
            yr_unique = np.unique(yr_non_zero)
            if len(yr_unique) <= n_bins:
                # Use actual unique values as bins
                self.yr_bins = np.concatenate([[0], yr_unique])
            else:
                # Use percentiles
                self.yr_bins = np.percentile(yr_non_zero, np.linspace(0, 100, n_bins + 1))
                self.yr_bins[0] = 0
        else:
            self.yr_bins = np.array([0, 1, 2, 3, 4, 5])  # Default bins
            
        return self.yq_bins, self.yr_bins
    
    def binize_targets(self, yq_values: np.ndarray, yr_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert continuous values to bins"""
        if self.yq_bins is None or self.yr_bins is None:
            raise ValueError("Bins must be created first using create_bins method")
            
        yq_binned = np.digitize(yq_values, self.yq_bins) - 1
        yr_binned = np.digitize(yr_values, self.yr_bins) - 1
        
        # Ensure within bounds
        yq_binned = np.clip(yq_binned, 0, len(self.yq_bins) - 2)
        yr_binned = np.clip(yr_binned, 0, len(self.yr_bins) - 2)
        
        return yq_binned, yr_binned
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw data"""
        df = df.copy()
        
        # Date features
        df['depart_year'] = df['departDate'] // 10000
        df['depart_month'] = (df['departDate'] % 10000) // 100
        df['depart_day'] = df['departDate'] % 100
        
        df['return_year'] = df['returnDate'] // 10000
        df['return_month'] = (df['returnDate'] % 10000) // 100
        df['return_day'] = df['returnDate'] % 100
        
        # Trip characteristics
        df['is_round_trip'] = (df['returnDate'] > 0).astype(int)
        df['total_duration'] = df['inboundDuration'].fillna(0) + df['outboundDuration'].fillna(0)
        
        # Advance purchase categories
        df['advance_purchase_cat'] = pd.cut(df['advancePurchase'], 
                                          bins=[-1, 7, 14, 30, 60, 365], 
                                          labels=['0-7days', '8-14days', '15-30days', '31-60days', '60+days'])
        
        # Length of stay categories
        df['length_of_stay_cat'] = pd.cut(df['lengthOfStay'], 
                                        bins=[-1, 3, 7, 14, 30, 365], 
                                        labels=['0-3days', '4-7days', '8-14days', '15-30days', '30+days'])
        
        # Price features
        df['amount_per_day'] = df['totalAmount'] / (df['lengthOfStay'] + 1)  # +1 to avoid division by zero
        
        # Route complexity
        df['is_same_carrier'] = (df['inboundOperatingCarrier'] == df['outboundOperatingCarrier']).astype(int)
        df['route_complexity'] = (df['inboundOriginAirportCode'] != df['outboundDestinationAirportCode']).astype(int)
        
        return df
    
    def preprocess_features(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Preprocess features for modeling"""
        # Select important columns for modeling
        categorical_cols = [
            'originCityCode', 'destinationCityCode', 'currencyCode',
            'validatingCarrier', 'inboundOperatingCarrier', 'outboundOperatingCarrier',
            'advance_purchase_cat', 'length_of_stay_cat'
        ]
        
        numerical_cols = [
            'advancePurchase', 'lengthOfStay', 'totalAmount',
            'inboundDuration', 'outboundDuration', 'depart_year', 'depart_month',
            'is_round_trip', 'total_duration', 'amount_per_day', 'is_same_carrier', 'route_complexity'
        ]
        
        # Handle categorical variables
        processed_data = []
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    # Convert to string and handle nulls
                    col_data = df[col].astype(str).fillna('Unknown')
                    encoded = self.label_encoders[col].fit_transform(col_data)
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        known_classes = set(self.label_encoders[col].classes_)
                        col_data = df[col].astype(str).fillna('Unknown')
                        col_data_mapped = col_data.apply(lambda x: x if x in known_classes else 'Unknown')
                        encoded = self.label_encoders[col].transform(col_data_mapped)
                    else:
                        encoded = np.zeros(len(df))
                processed_data.append(encoded.reshape(-1, 1))
        
        # Handle numerical variables
        numerical_data = df[numerical_cols].fillna(0).values
        
        if fit:
            numerical_data = self.scaler.fit_transform(numerical_data)
        else:
            numerical_data = self.scaler.transform(numerical_data)
        
        # Combine all features
        if processed_data:
            categorical_data = np.hstack(processed_data)
            features = np.hstack([categorical_data, numerical_data])
        else:
            features = numerical_data
            
        return features

def load_and_combine_data(data_dir: str = 'yqyr_data') -> pd.DataFrame:
    """Load and combine all parquet files"""
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    
    dfs = []
    print(f"Loading {len(data_files)} data files...")
    
    for file in tqdm(data_files, desc="Loading files"):
        file_path = os.path.join(data_dir, file)
        df = pd.read_parquet(file_path)
        dfs.append(df)
        print(f"Loaded {file}: {df.shape[0]:,} rows")
    
    print("Combining data...")
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataset shape: {combined_df.shape}")
    
    return combined_df

def prepare_targets(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare target variables (fill NaN with 0)"""
    yq_values = df['tax_yq_amount'].fillna(0).values.astype(np.float64)
    yr_values = df['tax_yr_amount'].fillna(0).values.astype(np.float64)
    
    yq_zero_mask = (yq_values == 0)
    yr_zero_mask = (yr_values == 0)
    
    print(f"YQ target - Zero values: {yq_zero_mask.sum():,} ({yq_zero_mask.mean()*100:.1f}%)")
    print(f"YR target - Zero values: {yr_zero_mask.sum():,} ({yr_zero_mask.mean()*100:.1f}%)")
    
    return yq_values, yr_values

def train_baseline_model(target_name: str = 'both'):
    """Train baseline Random Forest model"""
    
    print("="*60)
    print("BASELINE RANDOM FOREST MODEL TRAINING")
    print("="*60)
    
    # Load data
    print("\n1. LOADING DATA")
    df = load_and_combine_data()
    
    # Prepare targets
    print("\n2. PREPARING TARGETS")
    yq_values, yr_values = prepare_targets(df)
    
    # Initialize preprocessor
    print("\n3. PREPROCESSING")
    preprocessor = YQYRPreprocessor()
    
    # Feature engineering
    print("Engineering features...")
    df_processed = preprocessor.engineer_features(df)
    
    # Create classification bins
    print("Creating classification bins...")
    yq_bins, yr_bins = preprocessor.create_bins(yq_values, yr_values, n_bins=5)
    yq_classes, yr_classes = preprocessor.binize_targets(yq_values, yr_values)
    
    print(f"YQ bins: {yq_bins}")
    print(f"YR bins: {yr_bins}")
    print(f"YQ class distribution: {np.bincount(yq_classes)}")
    print(f"YR class distribution: {np.bincount(yr_classes)}")
    
    # Prepare features
    print("Processing features...")
    X = preprocessor.preprocess_features(df_processed, fit=True)
    print(f"Feature matrix shape: {X.shape}")
    
    # Feature selection to reduce computational load
    print("Selecting top features...")
    n_features = min(50, X.shape[1])  # Limit to 50 features for efficiency
    preprocessor.feature_selector = SelectKBest(f_classif, k=n_features)
    X_selected = preprocessor.feature_selector.fit_transform(X, yq_classes)
    print(f"Selected features shape: {X_selected.shape}")
    
    # Train models
    models = {}
    
    if target_name in ['yq', 'both']:
        print("\n4. TRAINING YQ MODEL")
        print("Splitting data...")
        # Use tqdm for progress bar during data splitting
        with tqdm(total=1, desc="Splitting YQ data", leave=False) as pbar:
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, yq_classes, test_size=0.2, random_state=42, stratify=yq_classes
            )
            pbar.update(1)
        
        print(f"YQ data split into X_train: {X_train.shape}, X_test: {X_test.shape}")
        
        print("Training Random Forest for YQ...")
        rf_yq = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=1000,
            min_samples_leaf=500,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        start_time = time.time()
        rf_yq.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"YQ model training completed in {training_time:.2f} seconds")
        
        # Evaluate
        y_pred = rf_yq.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"YQ Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        models['yq'] = rf_yq
    
    if target_name in ['yr', 'both']:
        print("\n5. TRAINING YR MODEL")
        print("Splitting data...")
        # Use tqdm for progress bar during data splitting
        with tqdm(total=1, desc="Splitting YR data", leave=False) as pbar:
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, yr_classes, test_size=0.2, random_state=42, stratify=yr_classes
            )
            pbar.update(1)
        print(f"YR data split into X_train: {X_train.shape}, X_test: {X_test.shape}")
        
        print("Training Random Forest for YR...")
        rf_yr = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=1000,
            min_samples_leaf=500,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        start_time = time.time()
        rf_yr.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"YR model training completed in {training_time:.2f} seconds")
        
        # Evaluate
        y_pred = rf_yr.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"YR Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        models['yr'] = rf_yr
    
    # Save models and preprocessor
    print("\n6. SAVING MODELS")
    os.makedirs('models', exist_ok=True)
    
    with open('models/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    for name, model in models.items():
        with open(f'models/rf_{name}_model.pkl', 'wb') as f:
            pickle.dump(model, f)
    
    print("Models saved successfully!")
    print("="*60)

if __name__ == "__main__":
    train_baseline_model() 