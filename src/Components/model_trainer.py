import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params):

        try:
            report = {}

            for model_name, model in models.items():

                logging.info(f"Training {model_name}")

                para = params[model_name]

                gs = GridSearchCV(
                    estimator=model,
                    param_grid=para,
                    cv=3,
                    scoring='r2',
                    n_jobs=-1
                )

                gs.fit(X_train, y_train)

                # Best parameters
                model.set_params(**gs.best_params_)

                # Train model
                model.fit(X_train, y_train)

                # Predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Scores
                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)

                logging.info(
                    f"{model_name} -> Train Score: {train_model_score}"
                )

                logging.info(
                    f"{model_name} -> Test Score: {test_model_score}"
                )

                report[model_name] = test_model_score

            return report

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_training(self, train_array, test_array):

        try:
            logging.info("Splitting training and testing data")

            # Split data
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Models
            models = {

                "Random Forest": RandomForestRegressor(),

                "Decision Tree": DecisionTreeRegressor(),

                "Gradient Boosting": GradientBoostingRegressor(),

                "Linear Regression": LinearRegression(),

                "KNeighbors Regressor": KNeighborsRegressor(),

                "XGBoost Regressor": XGBRegressor(),

                "CatBoost Regressor": CatBoostRegressor(
                    verbose=False
                ),

                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Hyperparameters
            params = {

                "Decision Tree": {
                    'criterion': [
                        'squared_error',
                        'friedman_mse',
                        'absolute_error',
                        'poisson'
                    ]
                },

                "Random Forest": {
                    'n_estimators': [
                        8, 16, 32, 64, 128, 256
                    ]
                },

                "Gradient Boosting": {
                    'learning_rate': [
                        .1, .01, .05, .001
                    ],
                    'subsample': [
                        0.6, 0.7, 0.75,
                        0.8, 0.85, 0.9
                    ],
                    'n_estimators': [
                        8, 16, 32, 64,
                        128, 256
                    ]
                },

                "Linear Regression": {},

                "KNeighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': [
                        'uniform',
                        'distance'
                    ]
                },

                "XGBoost Regressor": {
                    'learning_rate': [
                        .1, .01, .05, .001
                    ],
                    'n_estimators': [
                        8, 16, 32, 64,
                        128, 256
                    ]
                },

                "CatBoost Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [
                        0.01, 0.05, 0.1
                    ],
                    'iterations': [
                        30, 50, 100
                    ]
                },

                "AdaBoost Regressor": {
                    'learning_rate': [
                        .1, .01, 0.5, .001
                    ],
                    'n_estimators': [
                        8, 16, 32, 64,
                        128, 256
                    ]
                }
            }

            logging.info("Model evaluation started")

            model_report = self.evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params
            )

            # Best model score
            best_model_score = max(
                sorted(model_report.values())
            )

            # Best model name
            best_model_name = list(
                model_report.keys()
            )[
                list(model_report.values()).index(
                    best_model_score
                )
            ]

            best_model = models[best_model_name]

            logging.info(
                f"Best model found: {best_model_name}"
            )

            logging.info(
                f"Best model score: {best_model_score}"
            )

            # Validation
            if best_model_score < 0.6:
                raise CustomException(
                    "No best model found",
                    sys
                )

            # Train best model
            best_model.fit(X_train, y_train)

            # Save model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(
                "Best model saved successfully"
            )

            # Final prediction
            predicted = best_model.predict(X_test)

            r2_square = r2_score(
                y_test,
                predicted
            )

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)