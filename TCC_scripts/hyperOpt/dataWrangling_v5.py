import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime

##################################################################
#
#                      dataInget_csv
#
###################################################################

#bucket='bnr-bucket-01'
#file_key = 'EURUSD_Candlestick_1_M_BID_06.03.2021-11.03.2021.csv'
def dataIngest_csv(bucket, file_key):
    s3uri = 's3://{}/{}'.format(bucket, file_key)

    data = pd.read_csv(s3uri, header = 0)
    
    return data
###################################################################
#
#                     dataTransform
#
###################################################################
def convertToDatetime(s):
    match = re.search(r'\d{2}.\d{2}.\d{4} \d{2}:\d{2}', s)
    s_substring = match.group()

    # Create date object in given time format yyyy-mm-dd
    s_datetime = datetime.strptime(s_substring, '%d.%m.%Y %H:%M')

    return s_datetime

def dataTransform(data):
    data=data.drop(['Open', 'High', 'Low', 'Volume'], axis='columns', inplace=False)
    
    data['Gmt time'] = data["Gmt time"].apply(lambda x: convertToDatetime(x))
    data = data.rename(columns = {'Gmt time': 'datetime','Close': 'bidclose'}, inplace = False)
    
    data = data.set_index('datetime')
    
    return data

###################################################################
#
#                     dataCleaning
#
###################################################################

def dataClean(data):
    data = data.loc["2021-03-07 22:00:00":"2021-03-12 21:59:00"]
    
    return data
