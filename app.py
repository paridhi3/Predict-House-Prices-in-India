from src.HousePriceIndia.logger import logging
from src.HousePriceIndia.exception import CustomException
from src.HousePriceIndia.components.data_ingestion import DataIngestionConfig, DataIngestion
from src.HousePriceIndia.components.data_transformation import DataTransformationConfig, DataTransformation
from src.HousePriceIndia.components.model_trainer import ModelTrainerConfig, ModelTrainer

import sys

if __name__ == "__main__":
    logging.info("Execution started")
    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path, raw_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path, raw_data_path)

        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)