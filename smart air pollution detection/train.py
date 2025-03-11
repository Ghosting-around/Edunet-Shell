import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Check if the file exists
file_path = "data/city_day.csv"
if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' does not exist.")
    exit()

# Load the dataset
data = pd.read_csv(file_path)

# Drop rows with missing target values (PM2.5)
data = data.dropna(subset=["PM2.5"])

# Define features and target
X = data[["City", "NO2", "NO", "NOx", "PM10", "AQI"]]  # Features
y = data["PM2.5"]  # Target

# Identify categorical and numerical columns
categorical_cols = ["City"]
numerical_cols = ["NO2", "NO", "NOx", "PM10", "AQI"]

# Preprocessing for numerical data
numerical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="mean"),
        ),  # Fill missing values with the mean
        ("scaler", StandardScaler()),
    ]
)

# Preprocessing for categorical data
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=42)  # Use 100 trees

# Create a pipeline that preprocesses the data and then trains the model
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Save the model
joblib.dump(pipeline, "model.pkl")
print("Model saved as model.pkl")
