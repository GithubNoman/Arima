import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Initialize empty models dictionary
models = {}

# Define a route to receive new data through Flask API
@app.route('/new_data', methods=['POST'])
def predict():
    # Collect and clean data
    data = request.get_json()

    # Extract user_id and data from the JSON
    user_id = data['user_id']
    data = data['data']

    # Convert the data to a pandas dataframe
    data = pd.json_normalize(data)

    # Add user_id as a column
    data['user_id'] = user_id
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    print(data)

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
    result = {'user_id': user, 'prediction': result[user]}
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')