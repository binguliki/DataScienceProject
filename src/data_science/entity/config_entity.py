from dataclasses import dataclass
from pathlib import Path
from box import ConfigBox

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass
class DataValidationConfig:
    root_dir: Path
    unzip_data_dir: Path
    STATUS_FILE: Path
    all_schema: dict

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    
@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    target_column: str
    params: dict
    best_params_path: Path
    
@dataclass
class ModelEvaluationConfig:
    target_column: str
    root_dir: Path
    test_data_path: Path
    model_path: Path
    metric_file_name: Path
    best_params_path: ConfigBox

@dataclass
class PredictionConfig:
    columns: list
    preprocess_pipeline_path: Path
    model_path: Path