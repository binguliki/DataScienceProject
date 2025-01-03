## Component-1 => DataIngestion
import os
import urllib.request as request
from src.data_science import logger
from src.data_science.entity.config_entity import (DataIngestionConfig)
import zipfile

class DataIngestion:
    def __init__(self, config : DataIngestionConfig):
        self.config = config
    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename , headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists")
    
    def extract_zip_file(self):
        '''Extract the data from the zip file'''
        unzip_dir = self.config.unzip_dir
        os.makedirs(unzip_dir, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file , 'r') as file:
            file.extractall(unzip_dir)