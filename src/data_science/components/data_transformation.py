import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_science.entity.config_entity import (DataTransformationConfig)
from src.data_science import logger
import os

class DataTransformation:
    def __init__(self, config:DataTransformationConfig):
        self.config = config
    
    def train_test_splitting(self):
        data = pd.read_csv(self.config.data_path)

        train, test = train_test_split(data, test_size=0.2)
        train.to_csv(os.path.join(self.config.root_dir, 'train.csv'), index=False)
        test.to_csv(os.path.join(self.config.root_dir, 'test.csv'), index=False)

        logger.info("Splitted data into train and test sets")
        logger.info(f"Train data shape -> {train.shape}")
        logger.info(f"Test data shape -> {test.shape}")

        print(train.shape , test.shape)
    
    def transformation(self):
        pass # Can be done later