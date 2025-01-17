from src.data_science.entity.config_entity import PredictionConfig
from src.data_science.utils.common import load_bin
import pickle
import pandas as pd

class Predictor:
    def __init__(self, config: PredictionConfig):
        self.config = config
    
    def load_artifacts(self):
        self.model = load_bin(self.config.model_path)
        with open(self.config.preprocess_pipeline_path , 'rb') as file:
            self.pipe = pickle.load(file)
    
    def predict(self, data):
        df = pd.DataFrame([data] , columns=self.config.columns)
        processed_data = self.pipe.transform(df)
        prediction = self.model.predict(processed_data)

        return prediction