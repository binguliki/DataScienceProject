from src.data_science.entity.config_entity import DataValidationConfig
import pandas as pd

class DataValidation:
    def __init__(self, config:DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try: 
            validation_status = True
            data = pd.read_csv(self.config.unzip_data_dir)
            all_columns = list(data.columns)
            all_datatypes = list(data.dtypes)

            all_schema = self.config.all_schema

            with open(self.config.STATUS_FILE , 'w') as file:
                for col, data_type in zip(all_columns, all_datatypes):
                    dtype = all_schema.get(col , None)
                    if not dtype or dtype != data_type:
                        validation_status = False
                    else:
                        validation_status = True
                        file.write(f"{col} -> Validation Status: {validation_status}\n")

            return validation_status
        except Exception as e:
            raise e