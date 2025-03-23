import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
scaler=pickle.load(open("scaler.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    scaled_feature = scaler.transform(features)
    prediction=str(*model.predict(scaled_feature))
    return prediction

if __name__ == "__main__":
    app.run(debug=True)