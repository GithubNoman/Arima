import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima
from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import json

load_dotenv()

# Accessing environment variables
child_small = int(os.environ['AGE_5_6'])
child_middle = int(os.environ['AGE_7_8'])
child_older = int(os.environ['AGE_OTHER'])
#function for setting usage_time
def usage_time_predict(data, percent):
    predict_list = []
    for key,value in data.items():
                    value = value - (value * 0.1) 
                    temp = {'pkg_name':key, 'usage_time':int(value)}
                    predict_list.append(temp)
    return predict_list

with open('model_data.json') as f:
    model_data = json.load(f)
# Initialize FastAPI app
app = FastAPI()

class UsageData(BaseModel):
    user_id: str
    age : int
    data: list[dict]

# Initialize empty models dictionary
models = {}

# Define a route to receive new data through FastAPI API
@app.post("/new_data")
async def predict(usage: UsageData):
    # Collect and clean data
    # data = usage.dict()
    post_data = usage.dict()
    data = model_data
    child_age = data['age']
    # Extract user_id and data from the JSON
    user_id = post_data['user_id']
    data = data['data']

    # Convert the data to a pandas dataframe
    data = pd.json_normalize(data)

    # Add user_id as a column
    data['user_id'] = user_id
    data['Date'] = pd.to_datetime(data['Date']).dt.date

    child_age = post_data['age']

    post_data = post_data['data']
    post_data = pd.json_normalize(post_data)
    post_data['user_id'] = user_id
    post_data['Date'] = pd.to_datetime(post_data['Date']).dt.date
    # post_data.set_index('Date', inplace=True)
    data = pd.concat([data, post_data], axis=0)
    data = data[data['Date'] != data.iloc[0]['Date']]
    data.set_index('Date', inplace=True)


    # Group data by application name and update models
    result = {}
    for user, group in data.groupby('user_id'):
        if user not in models:
            models[user] = {}
        for appli, appli_group in group.groupby('pkg_name'):
            if appli not in models[user]:
                models[user][appli] = auto_arima(appli_group['usage_time'], trace=True, error_action='ignore', suppress_warnings=True)
            updated_data = pd.concat([appli_group, pd.DataFrame(models[user][appli].predict(n_periods=1), columns=['usage_time'], index=[group.index[-1]+pd.DateOffset(1)])], axis=0)
            models[user][appli].update(updated_data['usage_time'])
            forecast = models[user][appli].predict(n_periods=1, return_conf_int=True)
            if user not in result:
                result[user] = {}
            result[user][appli] = forecast[0].tolist()[0]
    # looping over the keys and values of result[user]
    predict_list = []
    sum_time = 0
    temp_time = 0
    for value in result[user].values():
        sum_time += value
    sum_time = int(sum_time)
    temp_time = sum_time - (sum_time * 0.1)
    if child_age == 5 or child_age == 6:
        if sum_time > child_small:
            if temp_time > child_small:
                predict_list = usage_time_predict(result[user], 0.1)
            else:
                predict_list = usage_time_predict(result[user], (temp_time/sum_time))
        else:
            predict_list = usage_time_predict(result[user], 0)

    if child_age == 7 or child_age == 8:
        if sum_time > middle_child:
            if temp_time > middle_small:
                predict_list = usage_time_predict(result[user], 0.1)
            else:
                predict_list = usage_time_predict(result[user], (temp_time/sum_time))
        else:
            predict_list = usage_time_predict(result[user], 0)

    if child_age == 9 or child_age == 10 or child_age == 11 or child_age == 12:
        if sum_time > other_child:
            if temp_time > other_child:
                predict_list = usage_time_predict(result[user], 0.1)
            else:
                predict_list = usage_time_predict(result[user], (temp_time/sum_time))
        else:
            predict_list = usage_time_predict(result[user], 0)

    result = {'user_id': user, 'prediction': predict_list}
            
    return result