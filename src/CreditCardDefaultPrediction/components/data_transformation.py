import os
import sys
import pandas as pd
from src.CreditCardDefaultPrediction.logger import logging
from src.CreditCardDefaultPrediction.exception import customexception
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import yeojohnson
from src.CreditCardDefaultPrediction.utlis.utils import save_object

class datatransformationconfig:
    preporcessor_path=os.path.join("artifacts","preprocessor.pkl")
class data_transformation:
    def __init__(self):
        self.prepocessor_path=datatransformationconfig()
    def data_transformation_preparation(self):
        try:
          preprocessor=StandardScaler()
          return preprocessor
        except Exception as e:
          logging.info("exception during occured at data tarnsformation preparation stage")
          raise customexception(e,sys)
    def data_transform_initiated(self,train_data,test_data):
       try:
          logging.info('trasnformation initiated')
          train_data=pd.read_csv(train_data)
          test_data=pd.read_csv(test_data)
          logging.info("read test and train data")
          target_column_name = 'default.payment.next.month'
          drop_columns = ['ID','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','default.payment.next.month']
          input_feature_train_df = train_data.drop(drop_columns,axis=1)
          target_feature_train_df=train_data[target_column_name]
          input_feature_test_df=test_data.drop(drop_columns,axis=1)
          target_feature_test_df=test_data[target_column_name]
          preprocessor_obj=self.data_transformation_preparation()
          input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
          logging.info('transformation has been complited')
          input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
          train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
          test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
          save_object(obj=preprocessor_obj,file_path=self.prepocessor_path.preporcessor_path)
          logging.info('save preprocessor object')
          logging.info("data tranformation has been complited")
          return (train_arr,test_arr)
          
       except Exception as e:
          logging.info("exception during occured at data tarnsformation initiation stage")
          raise customexception(e,sys)

          
            
            

    