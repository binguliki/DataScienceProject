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
        train.to_csv(os.path.join(self.config.root_dir, 'train.csv'), index=False)
        test.to_csv(os.path.join(self.config.root_dir, 'test.csv'), index=False)

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
            ('categorical-pipeline' , categorical_pipe , categorical_columns),
            ('numerical-pipeline' , numerical_pipe , numerical_columns)
        ], remainder="passthrough")

        processed_data = pipeline.fit_transform(data)
        new_categorical_columns = list(pipeline['categorical-pipeline'].get_feature_names_out()) 
        new_dataframe = pd.DataFrame(processed_data, columns=new_categorical_columns + numerical_columns + ['Trip_Price'])
        self.train_test_splitting(new_dataframe)

        with open(os.path.join(self.config.root_dir, 'pipeline.pkl'), 'wb') as file:
            pickle.dump(pipeline, file)

        logger.info(f'Saved the pipeline at {self.config.root_dir} ✅')