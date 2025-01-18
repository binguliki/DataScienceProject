from src.data_science import logger
from pathlib import Path
from src.data_science.entity.config_entity import ModelEvaluationConfig
from src.data_science.utils.common import load_bin, save_json
from sklearn.metrics import mean_squared_error
import pandas as pd
import mlflow
class ModelEvaluater:
    def __init__(self, config = ModelEvaluationConfig):
        self.config = config
    
    def evaluate(self):
        test_data = pd.read_csv(self.config.test_data_path)
        y_test = test_data[self.config.target_column].values
        x_test = test_data.drop(columns=[self.config.target_column]).values

        model = load_bin(Path(self.config.model_path))
        predictions = model.predict(x_test)
        error = mean_squared_error(y_test, predictions)

        metrics = {
            "Testing error" : error
        }
        save_json(Path(self.config.metric_file_name), metrics)

        mlflow.log_metric("Testing error ", error)
        logger.info(f"Model error on test data : {error}")