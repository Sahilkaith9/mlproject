import sys
from src.exceptions import CustomException
from src.logger import logging
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
import os
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from src.utils import load_obj

@dataclass
class DataTransformationConfig:
    preprocessor_path=os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
           numerical_columns = ["writing_score", "reading_score"]
           categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
           num_pipeline=Pipeline(
               [
                   ("imputer",SimpleImputer(strategy='median')),
                   ("Scaler",StandardScaler())
               ]
           )
           categorical_pipeline=Pipeline(
               [
                   ("imputer",SimpleImputer(strategy='most_frequent')),
                   ("ohe_hot_encoder",OneHotEncoder())
               ]
           )
           preprocessor=ColumnTransformer(
               [
                   ('Numerical_data',num_pipeline,numerical_columns),
                   ("Categorical_data",categorical_pipeline,categorical_columns)
               ]
           )
           logging.info("Implemented the pipeline")
           return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data")

            logging.info("Obtaining the preprocessor obj")

            preprocessor_obj=self.get_data_transformer_object()

            target_col_name="math_score"

            input_feature_train_df=train_df.drop(target_col_name,axis=1)
            input_feature_test_df=test_df.drop(target_col_name,axis=1)

            target_feature_train_df=train_df[target_col_name]
            target_feature_test_df=test_df[target_col_name]

            logging.info("Applying preprocessor object on train and test data")

            input_feature_train_df=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_df=preprocessor_obj.transform(input_feature_test_df)

            load_obj(
                file_path=self.data_transformation_config.preprocessor_path,
                obj=preprocessor_obj
            )

            train_arr=np.c_[
                input_feature_train_df,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[
                input_feature_test_df,np.array(target_feature_test_df)
            ]

            return train_arr,test_arr
            
        except Exception as e:

            raise CustomException(e,sys)
