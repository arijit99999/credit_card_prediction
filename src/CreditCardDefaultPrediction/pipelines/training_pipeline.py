from src.CreditCardDefaultPrediction.components.data_ingestion import data_ingestion
from src.CreditCardDefaultPrediction.components.data_transformation import data_transformation
from src.CreditCardDefaultPrediction.components.model_trainer import ModelTrainer
import os
import sys
from src.CreditCardDefaultPrediction.logger import logging
from src.CreditCardDefaultPrediction.exception import customexception


obj1=data_ingestion()
train_data,test_data=obj1.initiate_data_ingestion()

obj2=data_transformation()
train_arr,test_arr=obj2.data_transform_initiated(train_data,test_data)

obj3=ModelTrainer()
model=obj3.initate_model_training(train_arr,test_arr)

