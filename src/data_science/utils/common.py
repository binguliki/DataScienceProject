import os
import yaml
from src.data_science import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from box.exceptions import BoxValueError

@ensure_annotations # Use to ensure type annotations when using a function
def read_yaml(path_to_yaml : Path) -> ConfigBox:
    '''
        Reads Yaml files and returns ConfigBox object
        Args : 
            path_to_yaml (Path): path of the the yaml file
        Returns:
            ConfigBox: ConfigBox type
    '''

    try:
        with open(path_to_yaml) as file:
            content = yaml.safe_load(file)
            logger.info(f"Yaml file : {path_to_yaml} loaded successfully !!")
            return ConfigBox(content)
    except BoxValueError:
            logger.info(f"Yaml file : {path_to_yaml} is empty !!")
            raise ValueError(f"Yaml file is empty !!")
    except Exception as e:
            raise e

@ensure_annotations
def create_directories(path_to_directories : list, verbose = True):
    '''
        Creates directories
        Args:
            path_to_directories (list): path of various directories.
            verbose (bool): Log the creation of directories or not
        Returns: None
    '''
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at : {path}")

@ensure_annotations
def save_json(path: Path, payload: dict):
    '''
        Saves data in json format
        Args:
            path (Path): path of json file
            payload (dict): Data to be saved
        Returns: None
    '''
    with open(path, 'w') as file:
        json.dump(payload , file, indent=3)
    logger.info(f"Json file saved at : {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    '''
        Loads data from json file
        Args:
            path (Path): path to json file
        Returns:
            ConfigBox: ConfigBox type
    '''
    with open(path, 'r') as file:
        payload = json.load(file)
    logger.info(f"Json file loaded successfully from : {path}")
    payload = ConfigBox(payload)
    return payload

@ensure_annotations
def save_bin(path: Path, data: Any):
    '''
        Saves data in binary format
        Args:
            path (Path): path of binary file
            data (Any): Data to be saved
        Returns: None
    '''
    joblib.dump(data, path)
    logger.info(f"Binary file saved at : {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    '''
        Loads data from binary format
        Args:
            path (Path): path of binary file
        Returns: 
            Any : Object stored in the binary file
    '''
    data = joblib.load(path)
    logger.info(f"Json file loaded from : {path}")
    return data