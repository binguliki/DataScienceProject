import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from src.data_science.entity.config_entity import (DataTransformationConfig)
from src.data_science import logger
import os

class DataTransformation:
    def __init__(self, config:DataTransformationConfig):
        self.config = config
    
    def train_test_splitting(self, data):
        train, test = train_test_split(data, test_size=0.2)
        np.save(os.path.join(self.config.root_dir, 'train.npy'), train)
        np.save(os.path.join(self.config.root_dir, 'test.npy'), test)

        logger.info("Splitted data into train and test sets")
        logger.info(f"Train data shape -> {train.shape}")
        logger.info(f"Test data shape -> {test.shape}")

        print(train.shape , test.shape)
    
    def transformation(self):
        data = pd.read_csv(self.config.data_path)
        numerical_columns = data.select_dtypes(include='float64').columns.to_list()
        categorical_columns = data.select_dtypes(include='object').columns.to_list()

        numerical_columns.remove('Trip_Price') ## Target variable
        #Based on observation, Passenger count has discrete values that can be encoded as a one-hot-vector
        numerical_columns.remove('Passenger_Count')
        categorical_columns.append('Passenger_Count')

        data.dropna(subset=['Trip_Price'], inplace=True) # Remove rows with null output
        categorical_pipe = Pipeline([
            ('imputer' , SimpleImputer(strategy='most_frequent')),
            ('encoder' , OneHotEncoder(sparse_output=False))
        ])

        numerical_pipe = Pipeline([
            ('imputer' , SimpleImputer(strategy='mean')),
            ('scaler' , StandardScaler())
        ])

        pipeline = ColumnTransformer([
            ('column-pipeline' , categorical_pipe , categorical_columns),
            ('numerical-pipeline' , numerical_pipe , numerical_columns)
        ])

        processed_data = pipeline.fit_transform(data)
        self.train_test_splitting(processed_data)

        with open(os.path.join(self.config.root_dir, 'pipeline.pkl'), 'wb') as file:
            pickle.dump(pipeline, file)

        logger.info(f'Saved the pipeline at {self.config.root_dir} ✅')