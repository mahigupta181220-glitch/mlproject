import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.Components.data_transformation import DataTransformation
from src.Components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join(
        'artifacts',
        'train.csv'
    )

    test_data_path = os.path.join(
        'artifacts',
        'test.csv'
    )

    raw_data_path = os.path.join(
        'artifacts',
        'data.csv'
    )


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("Entered Data Ingestion Component")

        try:
            file_path = os.path.join(
                'notebook',
                'data',
                'stud.csv'
            )

            df = pd.read_csv(file_path)

            logging.info("Dataset read successfully")

            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.train_data_path
                ),
                exist_ok=True
            )

            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logging.info("Train Test Split initiated")

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

            logging.info("Ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    try:
        ingestion = DataIngestion()

        train_data, test_data = (
            ingestion.initiate_data_ingestion()
        )

        data_transformation = DataTransformation()

        train_arr, test_arr, _ = (
            data_transformation.initiate_data_transformation(
                train_data,
                test_data
            )
        )

        print("Data Transformation Completed")

        model_trainer = ModelTrainer()

        r2_score = model_trainer.initiate_model_training(
            train_arr,
            test_arr
        )

        print("Model Training Completed")
        print("R2 Score:", r2_score)

    except Exception as e:
        print(e)