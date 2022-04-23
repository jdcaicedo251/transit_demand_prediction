from tabnanny import verbose
import tensorflow as tf
import time

from data import unnormalize_predict
from utils import compile_and_fit
from time_measure import TicToc

import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class denseClass():
    def __init__(self, settings, station_name):
        self.aggregation = settings['aggregation']
        self.station_name = station_name

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

    def fit(self, window,  patience = 2):
        num_features = len(window.label_columns)
        neural_net = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(self.forecast_window*num_features,
            kernel_initializer=tf.initializers.zeros()),
            tf.keras.layers.Reshape([self.forecast_window, num_features])
            ])

        history = compile_and_fit(neural_net, window, patience=patience)
        self.fitted_model = neural_net
        
    def predict(self, window, train_mean, train_std, online = False):
        t = TicToc()
        model = self.fitted_model

        if online:
            for input, labels in window.test:
                normalized_prediction = model(input)
                prediction = unnormalize_predict(normalized_prediction, train_mean, train_std)

                #Debugging information
                logger.debug("Input shape for dense model is {}".format(input.shape))
                logger.debug("Output shape for dense model is {}".format(prediction.shape))
        else:
            normalized_prediction = model.predict(window.test)
            prediction = unnormalize_predict(normalized_prediction, train_mean, train_std)
        self.run_time = t.tocvalue()
        return prediction



        # # normalized_prediction = model.predict(window.test)
        # normalized_prediction = model(window.test)
        # prediction = unnormalize_predict(normalized_prediction, train_mean, train_std)
        # # prediction_shape = prediction.shape
        # # prediction = prediction.reshape((prediction_shape[0], prediction_shape[1]))

        # # station_dict = {}
        # # station_dict['name'] = self.station_name
        # # station_dict['run_time'] = run_time
        # # station_dict['prediction'] = prediction.tolist()
        # # logging.info('Prediction for station {} compleated. Time {:.2f} minutes'.format(self.station_name, run_time)) 
        # # return station_dict
        # return prediction