import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

Churn_Model = joblib.load(r'C:\Users\OlumayowaOyaleke\Desktop\DSML-Projects-1\Churn Analysis\Churn_Detection_Model_joblib')

# Importing Data
df_Train = pd.read_csv(r'C:\Users\OlumayowaOyaleke\Desktop\DSML-Projects-1\Churn Analysis\Train.csv',encoding='windows-1252')
df_Test = pd.read_csv(r'C:\Users\OlumayowaOyaleke\Desktop\DSML-Projects-1\Churn Analysis\Test.csv',encoding='windows-1252')
X_train = df_Train.copy()
X_train = X_train.drop("labels",axis=1)

scaler = MinMaxScaler()
scaler.fit(X_train)

def pre_processing_predict(data):
    transformed_data = scaler.transform(data)
    transformed_data = pd.DataFrame(transformed_data,columns=X_train.columns)
    Predictions = Churn_Model.predict(transformed_data)
    Predictions = pd.DataFrame(Predictions,columns=["result"])
    # result = ''
    # if Predictions[0] == 1:
    #     return "This is customer is laible to churn"
    #     #print("This is an Anomaly")
    # else:
    #     return "This is customer is going no where"
    # # return data 
    print(Predictions[Predictions["result"] == 1])
       
pre_processing_predict(df_Test)    

