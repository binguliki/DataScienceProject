from src.data_science.config.configuration import ConfigurationManager
from src.data_science.components.model_evaluation import ModelEvaluater

class ModelEvaluaterPipeline:
    def __init__(self):
        pass

    def initiate_model_evaluation(self):
        try: 
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluater = ModelEvaluater(model_evaluation_config)

            model_evaluater.evaluate()
        except Exception as e:
            raise e