from src.data_science.config.configuration import ConfigurationManager
from src.data_science.components.model_trainer import ModelTrainer

class ModelTrainerPipeline:
    def __init__(self):
        pass

    def initiate_model_trainer(self):
        try: 
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer = ModelTrainer(model_trainer_config)

            model_trainer.train()
        except Exception as e:
            raise e