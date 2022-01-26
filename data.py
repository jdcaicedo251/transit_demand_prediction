import pandas as pd 
import numpy as np 
import os 
import unicodedata
import tensorflow as tf

from utils import read_yaml

def read_settings():
    return read_yaml('settings.yaml')

def strip_accents(text):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.

    reference: https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string
    hexaJer cooment on Jul 24 2015. Edited Nov 23 2017. 
    """
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError): # unicode is a default on python 3 
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text).lower()

def read_data(settings):
    # start = time.time()
    input_path = settings['input']
    path = os.path.join(input_path['folder'], input_path['fname'])
    df = pd.read_csv(path, parse_dates = ['timestamp'])
    df.columns = [strip_accents(col) for col in df.columns]
    # end = time.time()
    # print('Elapsed time is {}'.format(end-start))
    return df 

def preprocess_data(settings):
    df = read_data(settings)
    # Time variables
    df['year'] = df.timestamp.dt.year
    df['month'] = df.timestamp.dt.month
    df['day'] = df.timestamp.dt.day
    df['hour'] = df.timestamp.dt.hour
    df['dayofweek'] = df.timestamp.dt.dayofweek 
    df['weekday'] = (df.dayofweek <= 4).astype(int)

    # Cleaning operation times 
    df = df[~df.hour.isin([0,1,2,3,23])]
    return df

def aggreagtion(df, aggregation = None):
    """
    Aggregates transactions by the given aggregation parameter. 
    
    Parameters:
    -----------
    - df: Pandas DataFrame, 
        Transactions by station. 
    - aggregation: str, default = None. 
        Aggregation interval. If none, it returns transactions every 15 mins. 
        One of ['hour','day','month']
    """
    # stations = set(df.columns) - set(['day', 'hour', 'weekday', 'year', 'month', 'dayofweek', 'time'])
    stations = df.columns.str.contains("\(")
    stations = df.columns[stations]

    if aggregation is None:
        return df.reset_index(drop = True) # Why drop index? 

    if aggregation == 'hour':
        groupby_list = ['year', 'month', 'day', 'dayofweek','weekday', 'hour']
    elif aggregation == 'day':
        groupby_list = ['year', 'month', 'day', 'dayofweek','weekday']
        df.drop(columns = ['hour'], inplace = True)
    elif aggregation == 'month':
        groupby_list = ['year', 'month']
        df.drop(columns = ['hour', 'day'], inplace = True)
    else:
        raise ValueError ('parameter {} not understood. Aggregation one of [None, hour, day, month]'.format(aggregation))

    #Groupby 
    # df1 = df.groupby(groupby_list).sum().reset_index() #Why dobble reset indexing? 
    # df1 = df1.reset_index(drop = True)

    #Data transformation (this should be last - after aggregating data)
    # Notice that this transformation is no longer available. Add it as an argument if I want to reveser this
    # if transformation == 'log':
    #     df1[stations] = df1[stations].transform(np.log1p)
    return df.groupby(groupby_list).sum().reset_index()

def train_test_data(settings):         
    aggregation = settings['aggregation']
    train_date = settings['train_date']
    train_date = (train_date['year'], train_date['month'], train_date['day'])

    df = preprocess_data(settings)
    df = aggreagtion(df, aggregation = aggregation)

    if aggregation == 'month':
        train_index = df[(df.year == train_date[0]) & (df.month == train_date[1])].index[0]

    else: 
        train_index = df[(df.year == train_date[0]) & (df.month == train_date[1]) & (df.day == train_date[2])].index[0]

    train = df[:train_index]
    test = df[train_index:]

    stations = train.columns.str.contains("\(")
    stations = train.columns[stations]

    return train, test


def tf_dataframe(settings, train, test):
    '''
    Parameters
    ------------
    settings: dict. File settings.yaml
    train, test: Pandas dataframe with data 
    '''
    assert train.name == test.name 

    steps_back = settings['cnn']['steps_back']
    steps_future = settings['cnn']['steps_future']

    #join df 
    df = pd.concat((train, test))
    train_idx = train.shape[0] - steps_back

    # Non-time series columns
    ts_cols = train.name
    os_cols = set(df.columns) - set([ts_cols])

    ## Window handling 
    column_indices = {name: i for i, name in enumerate(df.columns)}
    ts_idx = [column_indices[key] for key in [ts_cols]]
    os_idx = [column_indices[key] for key in os_cols]
    y_idx = [column_indices[key] for key in [ts_cols]]

    x = np.array(df)
    data_points = df.shape[0] - steps_back - steps_future + 1

    def partition(x, i):
        ts_slide = slice(i, i + steps_back)
        os_slide = slice(i + steps_back, i + steps_back + 1)
        y_slide = slice(i + steps_back, i + steps_back + steps_future)

        ts = x[ts_slide,ts_idx].flatten(order = 'F')
        os = x[os_slide,os_idx].flatten()
        inputs = np.hstack((ts,os))
        
        target = x[y_slide,y_idx].flatten()
        return inputs, target

    results = np.array([partition(x, i) for i in range(data_points)])

    inputs = np.stack(results[:,0]).astype('float32')
    target = np.stack(results[:,1]).astype('float32')
    dataset = tf.data.Dataset.from_tensor_slices((inputs,target))

    train = dataset.take(train_idx)
    test = dataset.skip(train_idx)#.take(pre_value)
    return train, test


def normalize(train, test):

    assert train.name == test.name 
    name = train.name
    
    #Normalize train data:
    train_time_series = train[name]
    mean_train, std_train = train_time_series.mean(), train_time_series.std()
    normalize_train_time_series = (train_time_series - mean_train)/(std_train)

    #Normalize test data:
    test_time_series = test[name]
    normalize_test_time_series = (test_time_series - mean_train)/(std_train)

    train[name] = normalize_train_time_series
    test[name] = normalize_test_time_series

    return train, test, mean_train, std_train

def unnormalize_predict(normalized_prediction, train_mean, train_std):
    return (normalized_prediction * train_std) + train_mean

# Unit Test
# settings = read_settings()
# train, test = train_test_data(settings)
# # print(test["(02000) cabecera autopista norte"])
# # print(train["(02000) cabecera autopista norte"])
# print(train)
# settings = read_settings()
# df = read_data(settings)
# print(df)
