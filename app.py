from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('random_forest.pkl', 'rb'))
app = Flask(__name__)
CORS(app)
@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    data = request.get_json()
    for k,v in data.items():
        data[k] = float(v)
    data = pd.Series(data)
    data = pd.DataFrame(data).T
    prediction = model.predict(data)
    return jsonify(prediction[0])

@app.route('/', methods=['GET'])
@cross_origin()
def serve():
    return jsonify("This is the backend for the Fish Classifier")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)