import time
import pmdarima as pm
import numpy as np

from data import unnormalize_predict
from time_measure import TicToc

import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class arimaClass():
    def __init__(self, settings, station_name):
        self.aggregation = settings['aggregation']
        self.station_name = station_name
        self.forecast_window = settings['forecast_window']

    def fit(self, train):
        start = time.time()
        
        #Model estimation
        time_series = train[self.station_name]
        exogenous_variables = train.drop([self.station_name], axis=1)
        arima_model = pm.auto_arima(y = time_series, x = exogenous_variables , 
                                    start_p=3, start_q=3, max_p=10, max_q=10, 
                                    seasonal=False,stepwise=True, suppress_warnings=False, 
                                    error_action='ignore', d = 0, maxiter = 1000)
        
        self.fitted_model = arima_model
        #Logging info
        end = time.time()
        time_elapsed = (end - start)/60
        logging.info('Model for station {} compleated. Time {:.2f} minutes'.format(self.station_name, time_elapsed)) 
        
    
    def predict(self, test):
        model = self.fitted_model
        cols = list(set(test.columns) -set([self.station_name]))
        
        prediction_list = []
        for index, row in test.iterrows():
            y = row[self.station_name]
            x = row[cols].values

            prediction = model.predict(n_periods = self.forecast_window, X = x)
            prediction_list.append(prediction)
            model.update(y, x) # Update time series with actual value for next prediction

        prediction = np.array(prediction_list)
        return prediction 

