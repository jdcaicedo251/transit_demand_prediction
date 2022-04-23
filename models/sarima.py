import os
import time
import pmdarima as pm
import joblib 
import numpy as np

from data import unnormalize_predict
from time_measure import TicToc

import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class sarimaClass():
    def __init__(self, settings, station_name):
        self.aggregation = settings['aggregation']
        self.station_name = station_name
        self.output = settings['output_folder']

        if self.aggregation == 'day':
            self.forecast_window = 7
        elif self.aggregation == 'hour':
            self.forecast_window = 24
        elif self.aggregation == 'month':
            self.forecast_window = 12
        elif self.aggregation == '15mins':
            self.forecast_window = 8
        else:
            raise KeyError ('aggregation parameter is one of [day, hour, month, 15 mins]')

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

        # Persist file
        # pickle_tgt = os.path.join(self.output, self.aggregation, 'sarima', 'models', 'train', self.station_name + '.pkl')
        # joblib.dump(sarima_model, pickle_tgt, compress=3)
            
        #Logging info
        end = time.time()
        time_elapsed = (end - start)/60
        logging.info('Model for station {} compleated. Time {:.2f} minutes'.format(self.station_name, time_elapsed)) 
        
    
    def predict(self, test, train_mean, train_std):
        t = TicToc()

        model = self.fitted_model
        cols = list(set(test.columns) -set([self.station_name]))
        
        prediction_list = []
        for index, row in test.iterrows():
            y = row[self.station_name]
            x = row[cols].values

            prediction = model.predict(n_periods = self.forecast_window, X = x)
            prediction_list.append(prediction)
            model.update(y, x, maxiter=5) # Update time series with actual value for next prediction

        normalized_prediction = np.array(prediction_list)
        prediction = unnormalize_predict(normalized_prediction, train_mean, train_std)

        # Persist the model as pkl for another future predictions. 
        # pickle_tgt = os.path.join(self.output, self.aggregation, 'sarima', 'models', 'test', self.station_name + '.pkl')
        # joblib.dump(model, pickle_tgt, compress=3)

        #Printing logging information
        self.run_time = t.tocvalue()
        # logging.DEBUG('Prediction for station {} compleated. Time {:.2f} minutes'.format(self.station_name, self.run_time)) 
        # end = time.time()
        # run_time = (end - start)/60
        # logging.info('Prediction for station {} compleated. Time {:.2f} minutes'.format(self.station_name, run_time)) 

        #Organize prediction in dictonary 
        # station_dict = {}
        # station_dict['name'] = self.station_name
        # station_dict['run_time'] = run_time
        # station_dict['prediction'] = prediction.tolist()
        return prediction