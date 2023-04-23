import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join("artifacts","preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        """
        Use for Data Transformation
        """
        
        try:
            discrete_num = ['Seat comfort', 'Departure/Arrival time convenient', 'Food and drink', 'Gate location', 'Inflight wifi service', 'Inflight entertainment', 'Online support', 'Ease of Online booking', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Cleanliness', 'Online boarding']
            conti_num = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
            oe = ["Class"]
            ohe = ['Gender', 'Customer Type', 'Type of Travel']
            
            discrete_num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy = 'most_frequent')),
    ('scalar', StandardScaler())
])
            conti_num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy = 'median')),
    ('scalar', StandardScaler())
])
            cat_ohe_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy = 'most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])
            cat_oe_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy = 'most_frequent')),
    ('oe', OrdinalEncoder())
])
            logging.info(f"Discrete columns: {discrete_num}")
            logging.info(f"Continuous columns: {conti_num}")
            logging.info(f"Categorical columns: {oe + ohe}")
            
            transformer = ColumnTransformer(transformers=
    [
    ('tnf1', discrete_num_pipeline, discrete_num),
    ('tnf2', conti_num_pipeline, conti_num),
    ('tnf3', cat_oe_pipeline, oe),
    ('tnf4', cat_ohe_pipeline, ohe)
], remainder = 'passthrough')
            
            return transformer
            
            
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Reading of train and test data completed")
            
            logging.info("Obtaining preprocessor object")
            
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = "satisfaction"
            # discrete_num = ['Seat comfort', 'Departure/Arrival time convenient', 'Food and drink', 'Gate location', 'Inflight wifi service', 'Inflight entertainment', 'Online support', 'Ease of Online booking', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Cleanliness', 'Online boarding']
            
            input_feature_train_df = train_df.drop(columns = [target_column_name], axis = 1)
            train_df[target_column_name] = train_df[target_column_name].replace('satisfied','1')
            train_df[target_column_name] = train_df[target_column_name].replace('dissatisfied','0')
            train_df[target_column_name] = train_df[target_column_name].astype('int64')
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns = [target_column_name], axis = 1)
            test_df[target_column_name] = test_df[target_column_name].replace('satisfied','1')
            test_df[target_column_name] = test_df[target_column_name].replace('dissatisfied','0')
            test_df[target_column_name] = test_df[target_column_name].astype('int64')
            target_feature_test_df = test_df[target_column_name]
            
            logging.info(f"Applying preprocessing object on training and testing dataframe.")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            logging.info(f"Saved preprocessing object.")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
    