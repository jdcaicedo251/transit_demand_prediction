import os

import tensorflow as tf
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
from utils import compile_and_fit

import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class cnnClass():
    def __init__(self, settings):
        self.forecast_window = settings['forecast_window']
        self.conv_widht = settings['steps_back']
        

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