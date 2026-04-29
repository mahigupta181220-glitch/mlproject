import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.Components.data_transformation import DataTransformation
from src.Components.data_transformation import  DataTransformationConfig


# ================= CONFIG =================
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


# ================= INGESTION =================
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered Data Ingestion component")

        try:
            file_path = os.path.join('notebook', 'data', 'stud.csv')

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found at {file_path}")

            df = pd.read_csv(file_path)
            logging.info("Dataset loaded successfully")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )

            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logging.info("Train-test split started")

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


# ================= MAIN =================
if __name__ == "__main__":
    try:
        ingestion = DataIngestion()

        train_path, test_path = ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()

        train_arr, test_arr, preprocessor_path = (
            data_transformation.initiate_data_transformation(
                train_path,
                test_path
            )
        )

        print("Train file saved at:", train_path)
        print("Test file saved at:", test_path)
        print("Preprocessor saved at:", preprocessor_path)

    except Exception as e:
        print(e)