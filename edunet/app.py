from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({"PM2.5 Prediction": prediction.tolist()})


if __name__ == "__main__":
    app.run(debug=True)
