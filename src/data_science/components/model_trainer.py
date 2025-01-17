import pandas as pd
import os
from src.data_science import logger
from src.data_science.config.configuration import ModelTrainerConfig
from src.data_science.utils.common import write_yaml
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)

        x_train = train_data.drop(columns=[self.config.target_column]).values
        y_train = train_data[self.config.target_column].values
        model = RandomForestRegressor()
        gscv = GridSearchCV(model, self.config.params, scoring="neg_mean_squared_error", cv=2)

        gscv.fit(x_train, y_train)
        logger.info("Model has been trained successfully !!")
        best_model = gscv.best_estimator_
        best_score = gscv.best_score_
        best_params ={ 
            "best_params" : gscv.best_params_
        }
        write_yaml(best_params, Path(self.config.best_params_path))
        logger.info("Saved best model parameters.")
        joblib.dump(best_model, os.path.join(self.config.root_dir, self.config.model_name))
        
        logger.info("Best model has been saved successfully !!")
        logger.info(f"Best training score : {abs(best_score)}")