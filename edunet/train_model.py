import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import pickle

# Load dataset
df = pd.read_csv("cleaned_data.csv")

# Drop unnecessary columns
df = df.drop(
    columns=[
        "stn_code",
        "location_monitoring_station",
        "agency",
        "sampling_date",
        "date",
    ]
)

# Check the shape of the dataset
print("Shape of the dataset:", df.shape)

# Check for missing values
print("Missing values in the dataset before filling:")
print(df.isnull().sum())

# Fill missing values in feature columns
df['rspm'] = df['rspm'].fillna(df['rspm'].mean())
df['spm'] = df['spm'].fillna(df['spm'].mean())

# Check for missing values after filling
print("Missing values in the dataset after filling feature columns:")
print(df.isnull().sum())

# Drop rows with missing target values
df = df.dropna(subset=["pm2_5"])

# Check for missing values in target variable
if df["pm2_5"].isnull().sum() > 0:
    print("Warning: Target variable 'pm2_5' contains missing values after dropping rows.")
else:
    print("No missing values in target variable 'pm2_5'.")

# Encode categorical features
categorical_cols = ["state", "location", "type"]
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Define features and target
X = df.drop(columns=["pm2_5"])  # PM2.5 is the target, but we don't delete it from df
y = df["pm2_5"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Optionally reduce the dataset size for testing
# X_train, y_train = X_train.sample(n=1000), y_train.sample(n=1000)

# Train model with fewer estimators for faster fitting
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# Save the model
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model training and evaluation completed successfully.")