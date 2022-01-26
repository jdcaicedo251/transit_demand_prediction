import IPython
import IPython.display
import tensorflow as tf
import time

from data import unnormalize_predict

import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class rnnClass():
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

    def fit(self, train, batch_size = 128):
        start = time.time()
        
        for feat, targ in train.take(1):
            input_layer = feat.shape[0]
            target_layer = targ.shape[0]
            hidden_neurons = int((input_layer + target_layer)/2)

        neural_net = tf.keras.Sequential([
                    tf.keras.layers.Dense(hidden_neurons, activation='sigmoid'), # Hidden layer
                    tf.keras.layers.Dense(target_layer)])

        neural_net.compile(loss=tf.losses.MeanSquaredError(),
                           optimizer=tf.optimizers.Adam(),
                           metrics=[tf.keras.metrics.MeanAbsoluteError(),  
                                    tf.metrics.RootMeanSquaredError()])

        train_dataset = train.batch(batch_size, num_parallel_calls = 3)
        neural_net.fit(train_dataset, epochs=30)
        IPython.display.clear_output()

        end = time.time()
        time_elapsed = (end - start)/60
        logging.info('Model for station {} compleated. Time {:.2f} minutes'.format(self.station_name, time_elapsed)) 

        self.fitted_model = neural_net

        # return neural_net
        
    def predict(self, test, train_mean, train_std, batch_size = 128):
        start = time.time()
        
        model = self.fitted_model
        test = test.batch(batch_size, num_parallel_calls = 3)
   
        normalized_prediction = model.predict(test)
        prediction = unnormalize_predict(normalized_prediction, train_mean, train_std)

        end = time.time()
        run_time = (end - start)/60

        station_dict = {}
        station_dict['name'] = self.station_name
        station_dict['run_time'] = run_time
        station_dict['prediction'] = prediction.tolist()
        logging.info('Prediction for station {} compleated. Time {:.2f} minutes'.format(self.station_name, run_time)) 
        return station_dict