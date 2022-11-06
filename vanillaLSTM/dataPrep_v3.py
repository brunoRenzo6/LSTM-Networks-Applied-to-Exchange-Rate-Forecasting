import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import numpy

##################################################################
#
#                      splitData
#
###################################################################


def splitData(df):
    training_size=int(len(df)*0.8)
    test_size=len(df)-training_size
    train_data, test_data=df.iloc[0:training_size,:], df.iloc[training_size:len(df),:1]

    print("train_data size:"+str(len(train_data)))
    print("test_test size:"+str(len(test_data)))
    
    return train_data, test_data

##################################################################
#
#                      normalizeData
#
###################################################################

def normalizeData(train_data, test_data, scaler):
    train_data=scaler.fit_transform(np.array(train_data).reshape(-1,1))
    test_data=scaler.transform(np.array(test_data).reshape(-1,1))
    
    return train_data, test_data, scaler

##################################################################
#
#                   create_dataset
#
###################################################################

def create_dataset(dataset, time_step=1):
    dataX=[]
    dataY=[]
    currentX=[]
    
    for i in range(len(dataset)):
        if(i>=time_step):
            a=dataset[i-time_step:i,0]
            dataX.append(a)
            dataY.append(dataset[i,0])
    
    return numpy.array(dataX), numpy.array(dataY)
