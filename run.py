import os
from models.arima import arimaClass
from models.sarima import sarimaClass
from models.rnn import rnnClass

from data import train_test_data, read_settings, normalize, tf_dataframe
from utils import save_json

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# #########################
# ## RUN ARIMA MODELS ####
# #########################
# directory = 'output/arima'
# data_directory = 'data'
# aggregation = 'day'
# # transformation = 'log'
# overwrite = False
# limit = None
# arima_models = arimaClass(directory, data_directory, aggregation)

# ## Train arima 
# arima_models.train_all(limit = limit, overwrite = overwrite)

# ## Predict
# arima_models.predict_all(overwrite=overwrite)


#########################
## RUN SARIMA MODELS ###
#########################
# directory = 'output/sarima'
# data_directory = 'data'
# aggregation = 'day'
# overwrite = False
# limit = None
# sarima_models = sarimaClass(directory, data_directory, aggregation)

# Train sarima 
# sarima_models.train_all(limit = limit, overwrite = overwrite)

## Predict sarima
# sarima_models.predict_all(overwrite=overwrite)
# sarima_models.prediction('(04107) escuela militar')
# sarima_models.fit_arima('(04107) escuela militar')

###########################
####### HELPER FUNCTIONS ##
###########################

def valid_train(train):
    """ Validates that training data is non-zero """
    s = train[train.name].sum()
    
    if s == 0:
        return False
    else: 
        return True

def run_model(model, train, test, train_mean, train_std):
    
    station_name = train.name
    path = os.path.join(OUTPUT , AGGREGATION, model, station_name + '.json' )

    if os.path.exists(path):
        logging.info('Results for model {} and station {} already exist'.format(model, station)) 

    else: 
        if model == 'rnn':
            train_tf, test_tf = tf_dataframe(settings, train, test)
            model_class = rnnClass(settings, station_name = station)
            model_class.fit(train_tf)
            prediction = model_class.predict(test_tf, train_mean, train_std)
                
        elif model == 'arima':
            model_class = arimaClass(settings, station_name = station)
            model_class.fit(train)
            prediction = model_class.predict(test, train_mean, train_std)

        elif model == 'sarima':
            model_class = sarimaClass(settings, station_name = station)
            model_class.fit(train)
            prediction = model_class.predict(test, train_mean, train_std)

        else:
            raise KeyError ('model {} not found'.format(model))
                
        save_json(path, prediction)

    return None

###########################
##### RUNNING MODELS ######
###########################

# Step 1 Read settings and data
settings = read_settings()
AGGREGATION = settings ['aggregation']
OUTPUT = settings['output_folder']
number_stations = settings['number_of_stations']

train, test = train_test_data(settings)

#Step 2. Select one station (Then we will iterate over Stations)
stations = list(train.columns[train.columns.str.contains("\(")])
if number_stations > 0:
    stations = stations[:number_stations]
    
cols = set(train.columns[~train.columns.str.contains("\(")]) - set(['year', 'month', 'day'])

for station in stations: 
    train_station = train[list(cols)+[station]]
    train_station.name = station

    test_station = test[list(cols)+[station]]
    test_station.name = station

    is_train_valid = valid_train(train_station)

    if is_train_valid:
        train_station, test_station, train_mean, train_std = normalize(train_station, test_station)

        [run_model(model, train_station, 
                   test_station, train_mean, 
                   train_std) for model in settings['models']]

    else:
        logging.info('Model estimation not posible for station {}. No train data available'.format(station)) 

