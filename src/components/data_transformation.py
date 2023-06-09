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
            discrete_num = ['Seat_comfort', 'Departure_Arrival_time_convenient', 'Food_and_drink', 'Gate_location', 'Inflight_wifi_service', 'Inflight_entertainment', 'Online_support', 'Ease_of_Online_booking', 'Onboard_service', 'Leg_room_service', 'Baggage_handling', 'Checkin_service', 'Cleanliness', 'Online_boarding']
            conti_num = ['Age', 'Flight_Distance', 'Departure_Delay_in_Minutes', 'Arrival_Delay_in_Minutes']
            oe = ["Class"]
            ohe = ['Gender', 'Customer_Type', 'Type_of_Travel']
            
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
            
            outlier_col = ['Flight_Distance', 'Departure_Delay_in_Minutes', 'Arrival_Delay_in_Minutes',"Onboard_service", "Checkin_service"]
            for i in outlier_col:
                q1, q3 = np.percentile(input_feature_train_df[i],[25,75])
                iqr = q3 - q1
                lower = q1 - 1.5*iqr
                higher = q3 + 1.5*iqr
                outlier = input_feature_train_df[input_feature_train_df[i]>higher].index.append(input_feature_train_df[input_feature_train_df[i]<lower].index)
                input_feature_train_df.drop(outlier, axis = 0, inplace = True)
                target_feature_train_df.drop(outlier, axis = 0, inplace = True)
            
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
    