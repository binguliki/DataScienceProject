from src.data_science.config.configuration import ConfigurationManager
from src.data_science.components.data_ingestion import DataIngestion

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_ingestion(self):
        config = ConfigurationManager()
        data_ingest_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingest_config)

        data_ingestion.download_file()
        data_ingestion.extract_zip_file()