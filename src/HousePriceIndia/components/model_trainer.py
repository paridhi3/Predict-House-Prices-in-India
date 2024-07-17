import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.HousePriceIndia.exception import CustomException
from src.HousePriceIndia.logger import logging
from src.HousePriceIndia.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mse = mean_squared_error(actual, pred)
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
                }
            
            model_report:dict = evaluate_models(X_train, y_train, X_test, y_test, models)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

             ## To get best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]            
            
            best_model = models[best_model_name]

            print(f"Best model:{best_model_name}")

            # model_names = list(params.keys())

            # actual_model=""

            # for model in model_names:
                # if best_model_name == model:
                    # actual_model = actual_model + model

            # best_params = params[actual_model]

            # mlflow

            # mlflow.set_registry_uri("https://dagshub.com/krishnaik06/mlprojecthindi.mlflow")
            # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # with mlflow.start_run():

            #     predicted_qualities = best_model.predict(X_test)

            #     (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)

            #     mlflow.log_params(best_params)

            #     mlflow.log_metric("rmse", rmse)
            #     mlflow.log_metric("r2", r2)
            #     mlflow.log_metric("mae", mae)


            #     # Model registry does not work with file store
            #     if tracking_url_type_store != "file":

            #         # Register the model
            #         # There are other ways to use the Model Registry, which depends on the use case,
            #         # please refer to the doc for more information:
            #         # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            #         mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
            #     else:
            #         mlflow.sklearn.log_model(best_model, "model")


            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_score = r2_score(y_test, predicted)
            return r2_score

        except Exception as e:
            raise CustomException(e,sys)