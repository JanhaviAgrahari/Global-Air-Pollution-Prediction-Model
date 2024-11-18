import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")

            # Transform features using the loaded preprocessor
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        country,
        co_aqi_value,
        ozone_aqi_value,
        no2_aqi_value,
        aqi_category
    ):
        self.country = country
        self.co_aqi_value = co_aqi_value
        self.ozone_aqi_value = ozone_aqi_value
        self.no2_aqi_value = no2_aqi_value
        self.aqi_category = aqi_category

    def get_data_as_data_frame(self):
        """
        This function creates a DataFrame from input data, aligned with the required features.
        """
        try:
            data = {
                "Country": [self.country],
                "CO AQI Value": [self.co_aqi_value],
                "Ozone AQI Value": [self.ozone_aqi_value],
                "NO2 AQI Value": [self.no2_aqi_value],
                "AQI Category": [self.aqi_category],
            }

            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)
