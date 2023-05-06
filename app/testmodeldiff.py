import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
from flask import Flask, request, jsonify

# Initialize FLASK app
flaskapp = Flask(__name__)

# Collect and clean data
data = pd.read_csv('usage_data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.dropna(inplace=True)

# Check for stationarity
def test_stationarity(timeseries):
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(data['usage_time'])

# Preprocessing
data_diff = data.groupby('pkg_name').diff(periods=1)
data_diff.dropna(inplace=True)
test_stationarity(data_diff['usage_time'])

# Auto ARIMA model
models = {}
for app, group in data.groupby('pkg_name'):
    models[app] = auto_arima(group['usage_time'], trace=True, error_action='ignore', suppress_warnings=True)
    models[app].fit(group['usage_time'])

# mae = mean_absolute_error(test_data, predicted_values)
# rmse = np.sqrt(mean_squared_error(test_data, predicted_values))

# print(f"MAE: {mae:.2f}")
# print(f"RMSE: {rmse:.2f}")
# def recommendation(forecast, updated_data, optimal_value):
#     if (forecast-optimal_value) < difference:
#         return "Decrease the usage time"
#     elif forecast > difference:
#         return "Increase the usage time"
#     else:
#         return "Usage time is optimal"

# Define a route to receive new data through FLASK API
@flaskapp.route('/new_data', methods=['POST'])
def predict():
    data = request.get_json()
    new_data = pd.DataFrame(data)
    new_data['Date'] = pd.to_datetime(new_data['Date'])
    new_data.set_index('Date', inplace=True)
    result = {}
    for app, group in new_data.groupby('pkg_name'):
        updated_data = pd.concat([data[data['pkg_name'] == app], group]).iloc[-14:]
        models[app].fit(updated_data['usage_time'])
        forecast = models[app].predict(n_periods=1, return_conf_int=True)
        result[app] = {
            "usage_time": forecast[0],
            "lower_bound": forecast[1][0][0],
            "upper_bound": forecast[1][0][1]
        }
    return jsonify(result)

if __name__ == '__main__':
    flaskapp.run(port=5000, debug=True)
