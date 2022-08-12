import os
import time
import tensorflow as tf
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

from data import unnormalize_predict
from utils import compile_and_fit
from time_measure import TicToc

import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class lstmClass():
    def __init__(self, settings):
        self.forecast_window = settings['forecast_window']


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
        
    def predict(self, window, online = False):
        model = self.fitted_model
        if online:
            for input, labels in window.test:
                prediction = model(input)

        else:
            prediction = model.predict(window.test)

        return prediction
