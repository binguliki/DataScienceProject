from src.data_science.components.data_transformation import DataTransformation
from src.data_science.config.configuration import ConfigurationManager
from pathlib import Path

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        try:
            with open(Path('artifacts/data_validation/status.txt')) as file:
                for row in file.readlines():
                    status = row.split(' ')[-1]
                    column = row.split(' ')[0]
                    if status == "True":
                        raise Exception(f'Column {column} has not been validated')
                    
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(data_transformation_config)

            data_transformation.transformation()
        except Exception as e:
            print(e)