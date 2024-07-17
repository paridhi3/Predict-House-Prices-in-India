from src.HousePriceIndia.logger import logging
from src.HousePriceIndia.exception import CustomException
from src.HousePriceIndia.components.data_ingestion import DataIngestionConfig, DataIngestion
# from HousePriceIndia
# from HousePriceIndia
import sys

if __name__ == "__main__":
    logging.info("Execution started")
    try:
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)