
import pandas as pd
import glob
from tqdm import tqdm
import gc
import numpy as np

def load_and_combine_data():
    """Load and combine all parquet files."""
    print("🔄 Loading and combining data...")
    data_files = glob.glob('yqyr_data/*.parquet')
    data_files.sort()
    print(f"Found {len(data_files)} data files")
    
    dfs = [pd.read_parquet(file) for file in tqdm(data_files, desc="Loading files")]
    combined_df = pd.concat(dfs, ignore_index=True)
    del dfs; gc.collect()
    
    print(f"✅ Combined dataset: {combined_df.shape}")
    return combined_df

def main():
    df = load_and_combine_data()
    
    # Add the 'yq' column as it's done in full_pipeline.py
    df['yq'] = df['tax_yq_amount'].fillna(0)

    carriers_to_check = ['AA', 'B6', 'DL']
    
    if 'validatingCarrier' in df.columns and 'yq' in df.columns:
        print("\nUnique YQ values for selected carriers:")
        for carrier in carriers_to_check:
            carrier_df = df[df['validatingCarrier'] == carrier]
            if not carrier_df.empty:
                unique_yq_values = carrier_df['yq'].unique()
                print(f"  Carrier {carrier}: {np.sort(unique_yq_values)}")
            else:
                print(f"  No data found for Carrier {carrier}")
    else:
        print("Required columns (\'validatingCarrier\' or \'yq\') not found in the dataset.")

    # Calculate and print percentage of rows for selected carriers
    if 'validatingCarrier' in df.columns:
        total_rows = len(df)
        print("\nPercentage of rows for selected carriers:")
        for carrier in carriers_to_check:
            carrier_rows = df[df['validatingCarrier'] == carrier].shape[0]
            percentage = (carrier_rows / total_rows) * 100 if total_rows > 0 else 0
            print(f"  Carrier {carrier}: {percentage:.2f}%")
    else:
        print("\'validatingCarrier\' column not found for percentage calculation.")

if __name__ == "__main__":
    main() 