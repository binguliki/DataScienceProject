from src.data_science.pipeline.data_ingestion import DataIngestionTrainingPipeline
from src.data_science.pipeline.data_validation import DataValidationTrainingPipeline
from src.data_science import logger

STAGE_NAME = 'Data Ingestion Stage'
try:
    logger.info(f">>>>>>>>>> Stage {STAGE_NAME} started <<<<<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.initiate_data_ingestion()
    logger.info(f">>>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<< \n\n")
except Exception as e:
    raise e

STAGE_NAME = 'Data Validation Stage'
try:
    logger.info(f">>>>>>>>>> Stage {STAGE_NAME} started <<<<<<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.initiate_data_validation()
    logger.info(f">>>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<< \n\n")
except Exception as e:
    raise e