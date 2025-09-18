from src.exceptions import CustomException
from src.logger import logging
import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join("artifacts","train.csv")
    test_data_path=os.path.join("artifacts","test.csv")
    raw_data_path=os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):


        logging.info("Entered data ingestion component")

        try:
           df = pd.read_csv('notebook/data/stud.csv')


           os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)
           logging.info("Initiating train test split") 
           train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

           train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
           test_set.to_csv(self.data_ingestion_config.test_data_path,index=True,header=True)
           df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=False)
           logging.info('Data ingestion completed')

           return self.data_ingestion_config.train_data_path,self.data_ingestion_config.test_data_path 
        except Exception as e:
            raise CustomException(e,sys)



if __name__=="__main__":
    obj=DataIngestion()
    train_path,test_path=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    train_arr,test_arr=data_transformation.initiate_data_transformation(train_path,test_path)
    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))

   
   
