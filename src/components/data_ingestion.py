# Will have all the code reladed to reading the data
# devide the dataset into train and test
# create a validation data

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.exception import CustomException
from src.logger import logging


from dataclasses import dataclass

from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method.")

        try:
            df = pd.read_csv(r'src\data\online_gaming_behavior_dataset.csv')
            logging.info("Dataset read completed.")

            logging.info(f"Creating directory: {os.path.dirname(self.ingestion_config.train_data_path)}")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            logging.info("Directory created or already exists.")
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("train and test data initiated.")
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion completed.")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    object=DataIngestion()  # Create an object of the DataIngestion class
    object.initiate_data_ingestion() # Call the initiate_data_ingestion method of that object


