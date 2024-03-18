from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import numpy as np
import pandas as pd
import os

model = pickle.load(open('models/random_forest.pkl', 'rb'))
app = Flask(__name__, static_folder='../build', static_url_path='')
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

@app.route('/')
@cross_origin()
def serve():
    return app.send_static_file(app.static_folder, 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)