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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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
def write_yaml(data: dict, path: Path):
    '''
        Writes into yaml file
        Args:
            data(dict) : data to be saved
            path(Path) : path of the file where it has to be saved.
    '''
    try:   
        if not (path.as_posix().endswith('yaml') or path.as_posix().endswith('.yml')):
            raise(f"Invalid file format !! Got: {path} Expected .yaml or .yml")
        
        with open(path , 'w') as file:
            yaml.safe_dump(data, file , indent=4, default_flow_style=False)
        logger.info(f"Dumped data into {path} successfully !!")
    except Exception as e:
        raise(e)

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

@ensure_annotations
def heat_map(data: list|np.ndarray , row_labels:list = [], col_labels:list = [], ax=None, 
            cbar_kw: dict=None, cbarlabel: str="", **kwargs):
    if ax is None:
        ax = plt.gca() # Creates axes if there is no plot

    if cbar_kw is None:
        cbar_kw = {} # if no keyword arguments are given for color-bar then convert it into empty dictionary

    # Plot the heatmap
    img = ax.matshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(img, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    
    # Set the ticks and their labels
    ax.set_xticks(range(len(row_labels)), labels= row_labels, rotation=30)
    ax.set_yticks(range(len(col_labels)), labels= col_labels)

    return img, cbar

@ensure_annotations
def annotate_heatmap(im, data : list = None, valfmt : str="{x:.2f}", **textkw):
    if not isinstance(data, (list , np.ndarray)):
        data = im.get_array()
    # Default text parameters which can be overwritten
    kw = dict(
        horizontalalignment = "center",
        verticalalignment = "center"
    )
    kw.update(**textkw)
    
    # Set the formatter
    valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
    
    # Write over the canvas at center of each axes
    n, m = data.shape
    for i in range(n):
        for j in range(m):
            im.axes.text(j, i, valfmt(data[i, j] , None), **kw)