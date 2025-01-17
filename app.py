from flask import Flask , request , render_template
import pickle
import json
import pandas as pd
from src.data_science.config.configuration import ConfigurationManager
from src.data_science.components.prediction import Predictor
app = Flask(__name__)

try: 
    config = ConfigurationManager()
    prediction_config = config.get_prediction_config()
    predictor = Predictor(prediction_config)
    predictor.load_artifacts()
except Exception as e:
    raise e

@app.route('/' , methods=['GET'])
def home():
    return render_template('index.html' , predicted_fare=None)

@app.route('/predict' , methods=['POST'])
def predict():
    data = request.form.to_dict().values()
    data = list(map(lambda x: int(x) if x.isdecimal() else x , data))
    try:
        prediction = predictor.predict(data)
    except Exception as e:
        raise e
    return render_template('index.html' , predicted_fare=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=8000)