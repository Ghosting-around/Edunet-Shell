from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from datetime import datetime

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

# Load the dataset (for displaying data)
data = pd.read_csv("data/city_day.csv")

# Convert the 'Date' column to datetime format
data["Date"] = pd.to_datetime(data["Date"])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get the city name and date range from the form
    city_name = request.form["city_name"]
    start_date = request.form["start_date"]
    end_date = request.form["end_date"]
    print(f"City name received: {city_name}")  # Log the city name
    print(f"Date range received: {start_date} to {end_date}")  # Log the date range

    # Verify that the CSV file is loaded correctly
    if data.empty:
        print("Error: The dataset is empty.")
        return jsonify({"error": "The dataset is empty."})

    # Convert start_date and end_date to datetime objects
    try:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Invalid date format. Please use YYYY-MM-DD."})

    # Filter data for the selected city and date range
    filtered_data = data[
        (data["City"] == city_name)
        & (data["Date"] >= start_date)
        & (data["Date"] <= end_date)
    ]

    if filtered_data.empty:
        print(
            f"No data found for city: {city_name} and date range: {start_date} to {end_date}"
        )  # Log the city name and date range
        return jsonify({"error": "No data found for the given city and date range."})

    # Prepare features for prediction
    features = filtered_data[["City", "NO2", "NO", "NOx", "PM10", "AQI"]]

    # Make predictions
    predictions = model.predict(features)
    filtered_data["Predicted_PM2.5"] = predictions

    # Convert the result to a dictionary
    result = filtered_data[
        ["City", "Date", "NO2", "NO", "NOx", "PM10", "AQI", "Predicted_PM2.5"]
    ].to_dict(orient="records")

    # Log the result for debugging
    print("Result being sent to frontend:", result)

    # Return the results as JSON
    return jsonify(result)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
