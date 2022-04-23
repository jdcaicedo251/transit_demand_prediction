import os
import shutil
import pandas as pd
import numpy as np
# from pytictoc import TicToc
import tensorflow as tf 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from time_measure import TicToc
from models.arima import arimaClass
from models.sarima import sarimaClass
from models.dense import denseClass
from models.cnn import cnnClass
from models.lstm import lstmClass

from data import train_test_data, read_settings, normalize, tf_dataframe, WindowGenerator
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

def create_path(path, replace = False):
    """ Create path if path is not defined. If replace is True, removes and create empty path"""
    try:
        os.makedirs(path)
    except FileExistsError:
        if replace:
            shutil.rmtree(path)
            os.makedirs(path)

def create_file_directory(settings, replace = False):
    for model in settings['models']:
        if static: 
                path = os.path.join(OUTPUT, AGGREGATION, model, 'static')
                create_path(path, replace)

        if online:
                path = os.path.join(OUTPUT, AGGREGATION, model, 'online')
                create_path(path, replace)

def prediction_dict(station, prediction, time):
    """ Saves a <station_name>.json file in the path directory. 
    
    Parameters:
    ------------
    path: 'str'. Directory path to save prediction 
    station_name: 'str'. Name of the station.
    time: float. Running time in secongs 
    prediction: array-like. Prediction array of the shape (n, OUT STEPS)
    """
    dict_ = {}
    dict_['name'] = station
    dict_['time'] = time
    dict_['prediction'] = prediction.tolist()
    return dict_

def run_static_model(model, train, test):
    
    station_name = train.name
    train_normalize, test_normalize, train_mean, train_std = normalize(train, test)
    # print("Statics Model >> Mean: {}, and Std {}".format(train_mean, train_std))
    
    path = os.path.join(OUTPUT , AGGREGATION,  model, 'static', station_name + '.json')

    if os.path.exists(path):
        logging.info('Results for model {} and station {} already exist. Skip'.format(model, station)) 

    else: 
        if model == 'dense':
            input_widht = 14
            OUT_STEPS = 7
            data = pd.concat((train_normalize, test_normalize))

            w = WindowGenerator(input_width=input_widht, label_width=OUT_STEPS, 
                                shift=OUT_STEPS, data = data, label_columns=[station_name], 
                                batch_size = 32, train_date=TRAIN_DATE)
            
            # train_tf, test_tf = tf_dataframe(settings, train, test)
            model_class = denseClass(settings, station_name = station)
            model_class.fit(w)
            prediction = model_class.predict(w, train_mean, train_std)
            # time = model_class.run_time

        elif model == 'cnn':
            input_widht = 14
            OUT_STEPS = 7
            data = pd.concat((train_normalize, test_normalize))

            w = WindowGenerator(input_width=input_widht, label_width=OUT_STEPS, 
                                shift=OUT_STEPS, data = data,label_columns=[station_name], 
                                batch_size = 32, train_date=TRAIN_DATE)
            
            # train_tf, test_tf = tf_dataframe(settings, train, test)
            model_class = cnnClass(settings, station_name = station)
            model_class.fit(w)
            prediction = model_class.predict(w, train_mean, train_std)

        elif model == 'lstm':
            input_widht = 14
            OUT_STEPS = 7
            data = pd.concat((train_normalize, test_normalize))

            w = WindowGenerator(input_width=input_widht, label_width=OUT_STEPS, 
                                shift=OUT_STEPS, data = data,label_columns=[station_name], 
                                batch_size = 32, train_date=TRAIN_DATE)
            
            # train_tf, test_tf = tf_dataframe(settings, train, test)
            model_class = lstmClass(settings, station_name = station)
            model_class.fit(w)
            prediction = model_class.predict(w, train_mean, train_std)
   
        elif model == 'arima':
            model_class = arimaClass(settings, station_name = station)
            model_class.fit(train_normalize)
            prediction = model_class.predict(test_normalize, train_mean, train_std)

        elif model == 'sarima':
            model_class = sarimaClass(settings, station_name = station)
            model_class.fit(train_normalize)
            prediction = model_class.predict(test_normalize, train_mean, train_std)

        else:
            raise KeyError ('model {} not found'.format(model))
        
    return np.array(prediction), model_class.run_time


def run_model_online(model, train, test):

    if model == 'arima':
            return None

    if model == 'sarima':
            return None

    data = pd.concat((train, test))
    train_index = data.index.get_loc(TRAIN_DATE)
    station_name = train.name
    input_widht = 14 #This should come from the settings file
    OUT_STEPS = 7 #This should come from the settings file 

    # data = data.iloc[:train_index + 20] ## Testing edge case ii
    predictions = []
    counter = 1
    for i in data.index[train_index: -OUT_STEPS]:# train_index + 3]: #FIX ME 
        logger.debug("Counter: {}".format(counter))
        logger.debug("Prediction date: {}".format(i))

        new_train_index = data.index.get_loc(i)
        new_train_date = data.index[new_train_index]
        df = data[:new_train_index + OUT_STEPS]

        train_data = df[:new_train_index][station_name]
        train_mean = train_data.mean()
        train_std = train_data.std()

        df_copy = df.copy(deep = True)
        df_copy[station_name] = (df[station_name] - train_mean)/train_std

        w = WindowGenerator(input_width=input_widht, label_width=OUT_STEPS, 
                                    shift=OUT_STEPS, data = df_copy,label_columns=[station_name], 
                                    batch_size = 32, train_date=new_train_date)

        if model == 'dense':
            model_class = denseClass(settings, station_name)

        if model == 'cnn':
            model_class = cnnClass(settings, station_name)

        if model == 'lstm':
            model_class = lstmClass(settings, station_name)

        model_class.fit(w)
        prediction = model_class.predict(w, train_mean, train_std, True)
        predictions.append(prediction)

        counter += 1

    return np.array(predictions), model_class.run_time

###########################
##### RUNNING MODELS ######
###########################

# Step 1 Read settings and data
settings = read_settings()

AGGREGATION = settings ['aggregation']
OUTPUT = settings['output_folder']
OUT_STEPS = settings['cnn']['steps_future']
TRAIN_DATE = settings['train_date']
# TRAIN_DATE = (DATE['year'], DATE['month'],DATE['day'])
REPLACE = settings['replace']

number_stations = settings['number_of_stations']
static = settings['static']
online = settings['online']
# train_date = settings['train_date']
train_df, test_df = train_test_data(settings)
create_file_directory(settings, REPLACE)

#Step 2. Select one station (Then we will iterate over Stations)
stations = list(train_df.columns[train_df.columns.str.contains("\(")])
if number_stations > 0:
    stations = stations[:number_stations]

    
cols = set(train_df.columns[~train_df.columns.str.contains("\(")]) # Exogenous variables


for station in stations: 
    train_station = train_df[list(cols)+[station]]
    train_station.name = station

    test_station = test_df[list(cols)+[station]]
    test_station.name = station

    is_train_valid = valid_train(train_station)

    if is_train_valid:

        for model in settings['models']:
            t = TicToc()
            logging.info('{} - {} model'.format(station, model))
                    
            if static: 
                station_path = os.path.join(OUTPUT, AGGREGATION, model, 'static', station + '.json')

                if os.path.exists(station_path):
                    logging.info('Results for model {} and station {} already exist. Skip ...'.format(model, station)) 
                
                else:
                    t.tic()
                    # logging.info('Static estimation and prediction...') 
                    predictions, time_ = run_static_model(model, train_station, test_station)
                    print(time_)
                    predictions = predictions.reshape(-1,OUT_STEPS)
                    predictions = prediction_dict(station, predictions, time_)
                    save_json(station_path, predictions)
                    t.toc('Static estimation took'.format(station), restart = True)

            if online:
                station_path = os.path.join(OUTPUT, AGGREGATION, model, 'online', station + '.json')
                
                if os.path.exists(station_path):
                    logging.info('Results for model {} and station {} already exist. Skip ...'.format(model, station)) 
                
                else:
                    # logging.info('Online estimation and prediction ...'.format(model, station)) 
                    try: 
                        t.tic()
                        predictions, time = run_model_online(model,train_station,test_station)
                        predictions = predictions.reshape(-1,OUT_STEPS)
                        predictions = prediction_dict(station, predictions, time)
                        save_json(station_path, predictions)
                        t.toc('Online estimation took'.format(station), restart = True)
                    except TypeError:
                        pass

    else:
        logging.info('Model estimation not posible for station {}. No train data available'.format(station))
