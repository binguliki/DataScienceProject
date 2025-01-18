import pandas as pd
import os
from src.data_science import logger
from src.data_science.config.configuration import ModelTrainerConfig
from src.data_science.utils.common import write_yaml
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import mlflow
from mlflow.models import infer_signature
from urllib.parse import urlparse

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def train(self):
        with mlflow.start_run(run_name=f"run-{self.runs + 1}"):
            train_data = pd.read_csv(self.config.train_data_path)
            
            y_train = train_data[self.config.target_column].values
            x_train = train_data.drop(columns=[self.config.target_column]).values

            signature = infer_signature(x_train, y_train)

            model = RandomForestRegressor()
            gscv = GridSearchCV(model, self.config.params, scoring="neg_mean_squared_error", cv=2)

            gscv.fit(x_train, y_train)
            logger.info("Model has been trained successfully !!")
            best_model = gscv.best_estimator_
            best_score = gscv.best_score_
            best_params ={ 
                "best_params" : gscv.best_params_
            }

            write_yaml(best_params, Path(os.path.join(self.config.root_dir, 'best_params.yaml')))
            mlflow.log_params(gscv.best_params_)
            logger.info("Saved best model parameters.")
            
            logger.info(f"Best training score : {abs(best_score)}")
            mlflow.log_metric("Training error", abs(best_score))

            url_type = urlparse(mlflow.get_tracking_uri()).scheme

            if url_type != "file":
                mlflow.sklearn.log_model(best_model, "model", signature=signature, registered_model_name = f"Best Model-{self.runs+1}")
            else:
                mlflow.sklearn.log_model(best_model, "model", signature=signature)

            joblib.dump(best_model, os.path.join(self.config.root_dir, self.config.model_name))
            logger.info("Best model has been saved successfully !!")