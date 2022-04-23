from tabnanny import verbose
import tensorflow as tf
# import time

from data import unnormalize_predict
from utils import compile_and_fit
from time_measure import TicToc

import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class cnnClass():
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

    def fit(self, window,  patience = 2):
        num_features = len(window.label_columns)
        conv_model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: x[:, -self.conv_widht:, :]),
        tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(self.conv_widht)),
        tf.keras.layers.Dense(self.forecast_window*num_features,
                          kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([self.forecast_window, num_features])])

        history = compile_and_fit(conv_model, window)
        self.fitted_model = conv_model
    
    # @tf.function(experimental_relax_shapes=True)
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