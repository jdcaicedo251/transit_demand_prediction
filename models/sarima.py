import time
import pmdarima as pm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class sarimaClass():
    def __init__(self, settings, station_name):
        self.aggregation = settings['aggregation']
        self.station_name = station_name
        self.forecast_window = settings['forecast_window']

    def fit(self, train):
        start = time.time()
        #Model estimation
        time_series = train[self.station_name]
        exogenous_variables = train.drop([self.station_name], axis=1)
        sarima_model = pm.auto_arima(y = time_series, x = exogenous_variables , 
                                    start_p=2, d=0, start_q=2, max_p=7, max_q=7,
                                    srtart_P=1, start_Q = 1, max_P = 4, max_Q = 4, 
                                    m = 7, seasonal=True, stepwise=True, 
                                    suppress_warnings=True, error_action='ignore', maxiter=200)

        self.fitted_model = sarima_model
            
        #Logging info
        end = time.time()
        time_elapsed = (end - start)/60
        logging.info('Model for station {} compleated. Time {:.2f} minutes'.format(self.station_name, time_elapsed)) 
        

    def predict(self, test):
        model = self.fitted_model
        cols = list(set(test.columns) -set([self.station_name]))
        
        prediction_list = []
        for _, row in test.iterrows():
            y = row[self.station_name]
            x = row[cols].values
            prediction = model.predict(n_periods = self.forecast_window, X = x)
            prediction_list.append(prediction)
            model.update(y, x, maxiter=5) # Update time series with actual value for next prediction

        prediction = np.array(prediction_list)
        return prediction