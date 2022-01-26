import os
import time
import pmdarima as pm
import joblib 
import numpy as np

from data import split_data
from utils import delete_files_folder, read_plk, save_json

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class arimaClass():
    def __init__(self, directory, data_directory, aggregation, transformation = None):
        self.directory = directory
        self.data_directory = data_directory
        self.aggregation = aggregation 
        self.transformation = transformation 

        if aggregation == 'day':
            self.forecast_window = 7
        elif aggregation == 'hour':
            self.forecast_window = 24
        elif aggregation == 'month':
            self.forecast_window = 12
        elif aggregation == '15mins':
            self.forecast_window = 8
        else:
            raise KeyError ('aggregation parameter is one of [day, hour, month, 15 mins]')

        train, test = split_data(self.data_directory, 
            aggregation= self.aggregation, 
            transformation = self.transformation)
        
        self.train = train 
        self.test = test
        self.stations = set(train.columns[train.columns.str.contains("\(")])

    def stationarity():
        return None

    def _standarize_train(self, name):
        s_raw = self.train[name]
        mean_ = s_raw.mean()
        std_ = s_raw.std()
        return (s_raw - mean_)/(std_)

    def _standarize_test(self, name):
        s_raw = self.train[name]
        test = self.test[name]
        mean_ = s_raw.mean()
        std_ = s_raw.std()
        return (test - mean_)/(std_)

    def _unstandarize_predict(self, name, x):
        s_raw = self.train[name]
        mean_ = s_raw.mean()
        std_ = s_raw.std()
        return (x * std_) + mean_

    def fit_arima(self, name):
        out_folder = self.directory
        aggregation = self.aggregation
        s = self._standarize_train(name)

        start = time.time()
        #Model estimation
        arima_model = pm.auto_arima(s, start_p=2, start_q=2, max_p=7, max_q=7, seasonal=False,
                        stepwise=True, suppress_warnings=True, error_action='ignore', d = 0, 
                        maxiter = 200)
        
        # Persist file
        pickle_tgt = os.path.join(out_folder, 'models', aggregation, 'train', name + '.pkl')
        joblib.dump(arima_model, pickle_tgt, compress=3)
        end = time.time()
        
        #logging info
        time_elapsed = (end - start)/60
        logging.info('Model for station {} compleated. Time {:.2f} minutes'.format(name, time_elapsed)) 
        del arima_model

    def train_all(self, limit = None, overwrite = False):
        aggregation = self.aggregation
        out_folder = os.path.join(self.directory , 'models', aggregation, 'train')
        train = self.train

        if overwrite:
            delete_files_folder(out_folder)
        
        stations = self.stations
        estimated_stations = os.listdir(out_folder)
        estimated_stations = set([x[:-4] for x in estimated_stations]) #removes extension in file name
        remaining_stations = stations - estimated_stations
        logging.info('{} stations do not have an estimated model. Starting estimation...'.format(len(remaining_stations))) 

        if limit is not None:
            remaining_stations = list(remaining_stations)[:limit]

        for name in remaining_stations:
            s = train[name]
            if s.sum() == 0:
                logging.info('No data available for station {} for the training period. No model estimation'.format(name)) 
            else:
                self.fit_arima(name)
    
    def predict(self, name):
        aggregation = self.aggregation
        input_folder = os.path.join(self.directory, 'models', aggregation, 'train', name + '.pkl')
        model = read_plk(input_folder)
        test = self._standarize_test(name)
        
        prediction_list = []
        ape_list = []
        for value in test:#.loc[start_value:]:
            prediction = model.predict(self.forecast_window)
            # prediction_list.append(prediction.tolist())
            prediction_list.append(prediction)
            model.update(value) # Update time series with actual value for next prediction
            ape = np.abs((value - prediction[0])/value)
            ape_list.append(ape)
            # print (ape)
        
        # Persist the model as pkl for another future predictions. 
        pickle_tgt = os.path.join(self.directory, 'models', aggregation, 'test', name + '.pkl')
        joblib.dump(model, pickle_tgt, compress=3)
        
        return np.array(prediction_list), np.array(ape_list)

    def predict_all(self, overwrite = False):
        directory = self.directory
        aggregation = self.aggregation
        out_folder = os.path.join(directory, 'results', aggregation)

        stations = self.stations

        if overwrite:
            delete_files_folder(out_folder)
        
        predicted_stations = os.listdir(out_folder)
        predicted_stations = set([x[:-5] for x in predicted_stations]) #removes extension in file name
        remaining_stations = stations - predicted_stations
        logging.info('{} stations do not have a prediction. Prediction starting...'.format(len(remaining_stations))) 

        for name in remaining_stations:
            logging.info('Initialize prediction for {}.'.format(name)) 
            try: 
                start = time.time()
                predict, ape = self.predict(name)
                end = time.time()
                logging.info('Finished prediction for {}. Time {:.2f} minutes'.format(name, (end-start)/60)) 

                final_prediction = self._unstandarize_predict(name, predict)
                final_prediction = final_prediction.tolist()
                ape = ape.tolist()
                dict_ = {}
                dict_['name'] = name
                dict_['run_time'] = (end-start)/60
                dict_['prediction'] = final_prediction
                dict_['one_period_ape'] = ape
                out_folder_path = os.path.join(out_folder, name + '.json')
                save_json(out_folder_path, dict_)
            except FileNotFoundError: 
                pass
                # logging.info('Station {}  not included - pass '.format(name))