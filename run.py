import os
import shutil
import sys
import pandas as pd
import numpy as np
import argparse
import tensorflow as tf 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from time_measure import TicToc
from models.arima import arimaClass
from models.sarima import sarimaClass
from models.dense import denseClass
from models.cnn import cnnClass
from models.lstm import lstmClass

from data import train_test_data, read_settings, min_max, unnormalize_min_max, WindowGenerator, train_index
from utils import save_json

import logging
logging.basicConfig(
    stream=sys.stdout, level=logging.INFO,
    format='%(asctime)s %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)


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

    
def create_path(path, replace=False):
    """ 
    Create path if path is not defined. 
    If replace is True, removes and create
    empty path
    """
    try:
        os.makedirs(path)
    except FileExistsError:
        if replace:
            shutil.rmtree(path)
            os.makedirs(path)

            
# def create_file_directory(settings, replace=False):
#     for model in settings['models']:
#         if static: 
#                 path = os.path.join(OUTPUT, AGGREGATION, model, 'static')
#                 create_path(path, replace)

#         if online:
#                 path = os.path.join(OUTPUT, AGGREGATION, model, 'online')
#                 create_path(path, replace)

                
def prediction_dict(station, prediction, estimation_time, simulation_time):
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
    
    if (type(estimation_time) == list) and (type(simulation_time) == list):
        dict_['estimation_time'] = estimation_time.tolist()
        dict_['simulation_time'] = simulation_time.tolist()
        
    else: 
        dict_['estimation_time'] = estimation_time
        dict_['simulation_time'] = simulation_time
    dict_['prediction'] = prediction.tolist()
    return dict_


def online_estimation(model, stations, test_limit=None):
    """ 
    Simulates a online training time series model
    
    Parameters: 
    ------------
    - model: str. model type e.g.: 'dense', 'cnn', 'lstm'
    - stations: list. List of station(s) to model 
    - limit: int. Maximum test predictions
    
    Return: 
    - prediction array. shape:(stations, predictions, forecast_window)
    
    """
    # Add one because shifting exogenous variables will remove one
    # Add LABEL_WIDTH because online training will remove it. 
    if test_limit:
        test_limit = test_limit + 1 + label_width
        
    train_df = train[stations + exog_vars]
    test_df = test[stations + exog_vars].iloc[:test_limit]
    
    normalized_data = min_max(train_df,
                              test_df,
                              stations)
    
    _train, _test, _min, _max = normalized_data
    
    df = pd.concat((_train, _test))
    df[exog_vars] = df[exog_vars].shift(-1)
    df = df.iloc[:-1,:]
    
    train_index_ = train_index(df, train_date)
    predictions = []
    estimation_time = []
    simulation_time = []
    for i in df.index[train_index_: -label_width]:
        idx_online = train_index(df, i)
        df_online = df[:idx_online + 1]
        
        w = WindowGenerator(input_width=input_width, 
                            label_width=label_width,
                            shift=label_width, 
                            batch_size=batch_size, 
                            train_date=i,
                            label_columns=stations,
                            data=df_online)
        
        if model == 'dense':
            model_class = denseClass(settings)
        elif model == 'cnn':
            model_class = cnnClass(settings)
        elif model == 'lstm':
            model_class = lstmClass(settings)
        
        t = TicToc()
        t.tic()
        model_class.fit(w)
        estimation_time_ = t.tocvalue()
        estimation_time.append(estimation_time_)
        
        t.tic()
        prediction = model_class.predict(w, True)
        simulation_time_ = t.tocvalue()
        simulation_time.append(simulation_time_)
        
        predictions.append(prediction)
    
    results = np.array(predictions)
    results = unnormalize_min_max(results, _min, _max)
    results = np.squeeze(results, axis = 1)
    results = np.moveaxis(results, 2, 0) #(stations, predictions, forecast_window)
    
    estimation_time = np.array(estimation_time)#.mean()
    simulation_time = np.array(simulation_time)#.mean()
    
    return results, estimation_time, simulation_time


def static_estimation(model, stations, test_limit=None):
    """ 
    Simulates a static time series model
    
    Parameters: 
    ------------
    - model: str. model type e.g.: 'dense', 'cnn', 'lstm'
    - stations: list. List of station(s) to model 
    - limit: int. Maximum test predictions
    
    Return: 
    - prediction array. shape:(stations, predictions, forecast_window)
    
    """
    # Add one because shifting exogenous variables will remove one
    if test_limit:
        test_limit+=1 
        
    train_df = train[stations + exog_vars]
    test_df = test[stations + exog_vars].iloc[:test_limit,:]
    
    normalize_data = min_max(train_df,
                             test_df,
                             stations)
    
    _train, _test, _min, _max = normalize_data
    
    df = pd.concat((_train, _test))
    df[exog_vars] = df[exog_vars].shift(-1)
    df = df.iloc[:-1,:]
    
    w = WindowGenerator(input_width=input_width, 
                        label_width=label_width,
                        shift=label_width, 
                        batch_size=batch_size, 
                        train_date=train_date,
                        label_columns=stations,
                        data=df)
    
    t = TicToc()
    
    if model == 'dense':
        model_class = denseClass(settings)
    elif model == 'cnn':
        model_class = cnnClass(settings)
    elif model == 'lstm':
        model_class = lstmClass(settings)
    elif model == 'arima':
        model_class = arimaClass(settings, stations[0])
        
        t = TicToc()
        t.tic()
        model_class.fit(_train)
        estimation_time = t.tocvalue(restart=False)
        
        t.tic()
        results = model_class.predict(_test) #shape (prediction days, forecast_window)
        simulation_time = t.tocvalue(restart=False)

        results = unnormalize_min_max(results, _min, _max)
        results = np.expand_dims(results, axis=0)
        return results, estimation_time, simulation_time
    elif model == 'sarima':
        model_class = sarimaClass(settings, stations[0])
        
        t = TicToc()
        t.tic()
        model_class.fit(_train)
        estimation_time = t.tocvalue(restart=False)
        
        t.tic()
        results = model_class.predict(_test) #shape (prediction days, forecast_window)
        simulation_time = t.tocvalue(restart=False)
        
        results = unnormalize_min_max(results, _min, _max)
        results = np.expand_dims(results, axis=0)
        return results, estimation_time, simulation_time
    
    t = TicToc()
    t.tic()
    model_class.fit(w)
    estimation_time = t.tocvalue(restart=False)
    
    t.tic()
    results = model_class.predict(w) # Normalized prediction 
    simulation_time = t.tocvalue(restart=False)
    
    results = unnormalize_min_max(results, _min, _max)
    results = np.moveaxis(results, 2, 0) #shape (stations, predictions, forecast_window)
    return results, estimation_time, simulation_time


def parse_args_and_settings(settings_file='settings.yaml'):

    settings = read_settings()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-o', '--online', action='store_true', help='online training')

    parser.add_argument(
        '-m', '--model', action='store', help='model e.g. cnn, lstm, rnn, arima, sarima')

    parser.add_argument(
        '-s', '--stations', action='store',  nargs='+', 
        help='station name(s). Multiple stations names can be passed')
    
    parser.add_argument(
        '-a', '--aggregation', action='store', 
        help='Input aggregation. e.g. "15-mins","hour", "day"')
    
    parser.add_argument(
        '-tl', '--testlimit', action='store', type = int,
        help='Number of max prediction. If none is define, it will run all prediction available')
    
    
    args = parser.parse_args()
    
    if args.online: 
        settings.update({'online':args.online})
        settings['training'] = 'online'
    else:
        settings.update({'online':args.online})
        settings['training'] = 'static'
    
    if args.model:
        settings.update({'models': args.model})

    if args.stations:
        settings['stations'] = args.stations
        settings.update({'multioutput':False})
        settings.update({'output':'single'})
    else:
        settings.update({'multioutput':True})
        settings.update({'output':'multioutput'})
        
    if args.aggregation:
        settings['aggregation'] = args.aggregation
        
    if args.testlimit:
        settings['test_limit'] = args.testlimit 
    else:
        settings['test_limit'] = None

    return settings


###########################
##### RUNNING MODELS ######
###########################
if __name__ == '__main__':
    settings = parse_args_and_settings()
    
    output_folder = settings['output_folder']
    aggregation = settings ['aggregation']
    label_width = settings['forecast_window']
    input_width = settings['steps_back']
    train_date = settings['train_date']
    replace = settings['replace']
    online = settings['online']
    model = settings['models']
    stations = settings['stations']
    training = settings['training']
    test_limit = settings['test_limit']
    multioutput = settings['multioutput']
    output = settings['output']
    batch_size = settings['batch_size']

    logger.info("Initilize Time Series Experiments")
    logger.info("Model: {}".format(model))
    logger.info("Aggregation: {}".format(aggregation))
    logger.info("Training: {}".format(training))
    logger.info("Output: {}".format(output))
    
    to_drop_stations = ['(40000) cable portal tunal',
                        '(40001) juan pablo ii',
                        '(40002) manitas',
                        '(40003) mirador del paraiso']
    
    train, test = train_test_data(settings)
    train = train.drop(columns= to_drop_stations)
    test = test.drop(columns= to_drop_stations)
    exog_vars = list(set(train.columns[~train.columns.str.contains("\(")]))
    list_stations = list(train.columns[train.columns.str.contains("\(")])

    if multioutput:
        stations = list_stations
        logger.info('Input Stations: {}'.format(len(stations)))
    else: 
        stations = settings['stations']
        logger.info('Input Station: {}'.format(stations))
        
    results_path = path = os.path.join(output_folder,
                                       aggregation,
                                       training,
                                       output,
                                       model)
    
    create_path(results_path, replace = replace)
    
    time_= TicToc()

    if online: 
        logger.info('Running in Online Mode')
        time_.tic()
        result, e_time, s_time = online_estimation(model, stations, test_limit = test_limit)
        time_.toc("Model estimation and simulation in ")
    else: 
        logger.info('Running in Static Mode')
        time_.tic()
        result, e_time, s_time = static_estimation(model, stations, test_limit = test_limit)
        time_.toc("Model estimation and simulation in ")
        
    #Save results
    logger.info("saving results")
    for station, prediction in zip(stations, result):
        p_dict = prediction_dict(station, prediction, e_time, s_time)
        path = os.path.join(results_path, station + '.json')
        save_json(path, p_dict)