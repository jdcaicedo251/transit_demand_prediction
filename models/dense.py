import os
import time
import tensorflow as tf
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


from utils import compile_and_fit

import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class denseClass():
    def __init__(self, settings):
        self.forecast_window = settings['forecast_window']


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
        
    def predict(self, window, online = False):
        model = self.fitted_model

        if online:
            for input, labels in window.test:
                prediction = model(input)

                #Debugging information
                logger.debug("Input shape for dense model is {}".format(input.shape))
                logger.debug("Output shape for dense model is {}".format(prediction.shape))
        else:
            prediction = model.predict(window.test)
        return prediction