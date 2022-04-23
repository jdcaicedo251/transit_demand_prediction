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

class lstmClass():
    def __init__(self, settings, station_name):
        self.aggregation = settings['aggregation']
        self.station_name = station_name
        self.conv_widht = settings['cnn']['steps_back']

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

    def fit(self, window, patience = 2):
        num_features = len(window.label_columns)
        lstm_model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dense(self.forecast_window*num_features,
            kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([self.forecast_window, num_features])
            ])

        history = compile_and_fit(lstm_model, window)
        self.fitted_model = lstm_model
        
    def predict(self, window, train_mean, train_std, online = False):
        t = TicToc()
        model = self.fitted_model
        if online:
            for input, labels in window.test:
                normalized_prediction = model(input)
                prediction = unnormalize_predict(normalized_prediction, train_mean, train_std)
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
        # return prediction 