from src.data_science.pipeline.data_ingestion import DataIngestionTrainingPipeline
from src.data_science.pipeline.data_validation import DataValidationTrainingPipeline
from src.data_science.pipeline.data_transformation import DataTransformationTrainingPipeline
from src.data_science.pipeline.model_trainer import ModelTrainerPipeline
from src.data_science.pipeline.model_evaluation import ModelEvaluaterPipeline
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

STAGE_NAME = 'Data Transformation Stage'
try:
    logger.info(f">>>>>>>>>> Stage {STAGE_NAME} started <<<<<<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.initiate_data_transformation()
    logger.info(f">>>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<< \n\n")
except Exception as e:
    raise e

STAGE_NAME = 'Model Training Stage'
try:
    logger.info(f">>>>>>>>>> Stage {STAGE_NAME} started <<<<<<<<<")
    model_trainer = ModelTrainerPipeline()
    model_trainer.initiate_model_trainer()
    logger.info(f">>>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<< \n\n")
except Exception as e:
    raise e

STAGE_NAME = 'Model Evaluation Stage'
try:
    logger.info(f">>>>>>>>>> Stage {STAGE_NAME} started <<<<<<<<<")
    model_evaluater = ModelEvaluaterPipeline()
    model_evaluater.initiate_model_evaluation()
    logger.info(f">>>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<< \n\n")
except Exception as e:
    raise e