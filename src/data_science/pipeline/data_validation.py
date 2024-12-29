from src.data_science.components.data_validation import DataValidation
from src.data_science.config.configuration import ConfigurationManager

STAGE_NAME = "Data Validation Stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_validation(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(data_validation_config)

        data_validation.validate_all_columns()