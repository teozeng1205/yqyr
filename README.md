# YR Tax Prediction Model

Clean, streamlined pipeline for predicting YR tax amounts from airline booking data.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline (reproduces all results)
python full_pipeline.py
```

## What it does

This project provides a machine learning pipeline to predict YR tax amounts. It loads and preprocesses airline booking data, trains an optimized model, and evaluates its performance.

## Results

- **94.6% predictions within $2** of actual YR tax amount
- **6.50 RMSE**
- **0.995 R²**
- **Only 11 features** for fast inference
- **305KB model** for production readiness

## Output Files

- `yr_final_model.txt` - Trained LightGBM model
- `feature_names.csv` - List of features used
- `model_results.txt` - Performance summary

## Key Features

The model uses 11 features, with carrier information (validating and operating carriers) being the most important. 