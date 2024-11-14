import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for creating the data transformation pipeline.
        """
        try:
            # Define ordinal categories and features for encoding
            ordinal_categories = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
            ordinal_features = ['CO AQI Category', 'Ozone AQI Category', 'NO2 AQI Category', 'AQI Category']

            # Numerical columns to standardize
            numerical_columns = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value']

            # Categorical column to one-hot encode
            categorical_columns = ['Country']

            # Pipeline for numerical features (only scaling as we've already dropped missing values)
            num_pipeline = Pipeline(steps=[
                ('scaler', StandardScaler(with_mean=False))
            ])

            # Pipeline for ordinal features
            ordinal_pipeline = Pipeline(steps=[
                ('ordinal_encoder', OrdinalEncoder(categories=[ordinal_categories] * len(ordinal_features), dtype=int))
            ])

            # Pipeline for categorical features (one-hot encoding)
            cat_pipeline = Pipeline(steps=[
                ('one_hot_encoder', OneHotEncoder(sparse_output=False, drop='first', dtype=int))
            ])

            # Combine all transformations
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('ordinal_pipeline', ordinal_pipeline, ordinal_features),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ], remainder='passthrough')

            logging.info(f"Preprocessor object created with numerical, ordinal, and categorical pipelines.")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Dropping columns with high correlation
            columns_to_drop = ['PM2.5 AQI Category', 'City', 'PM2.5 AQI Value']
            train_df.drop(columns=columns_to_drop, inplace=True)
            test_df.drop(columns=columns_to_drop, inplace=True)

            # Dropping rows with missing values in numerical and categorical columns
            for df in [train_df, test_df]:
                df.dropna(subset=['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'Country'], inplace=True)
                # Replace countries with fewer than 50 occurrences with 'Others'
                country_counts = df['Country'].value_counts()
                other_countries = country_counts[country_counts < 50].index
                df['Country'] = df['Country'].apply(lambda x: 'Others' if x in other_countries else x)

            logging.info("Dropped missing values and transformed 'Country' column for both train and test data")

            # Obtain the preprocessing pipeline
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'AQI Value'
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")

            # Apply transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed features with target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
