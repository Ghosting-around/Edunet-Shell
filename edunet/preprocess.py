import pandas as pd
import os

# Check if the file exists
file_path = "data.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")

try:
    # Load dataset with a different encoding
    df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
except Exception as e:
    raise Exception(f"Error reading {file_path}: {e}")

# Display first few rows
print(df.head())

# Handle missing values
df.ffill(inplace=True)

# Convert date columns if needed
if 'date' in df.columns:
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        raise Exception(f"Error converting 'date' column to datetime: {e}")

# Save cleaned data
output_file_path = "cleaned_data.csv"
df.to_csv(output_file_path, index=False)
print(f"Cleaned data saved to {output_file_path}")
