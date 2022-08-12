import pandas as pd 
import numpy as np 
import os 
import unicodedata
import tensorflow as tf
import holidays_co

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
    
    #Clean dataset
    df.columns = [strip_accents(col) for col in df.columns]
    stations = df.columns[df.columns.str.contains("\(")]
    df = df.groupby('timestamp')[stations].sum()
    df = df[df.index <= pd.Timestamp('2021-04-30 23:45:00')]
    # hour = df.index.hour
    # df = df[~hour.isin([0,1,2,3,23])]   
    ## There are some duplicate 15-mins intervals that need to be sum up. 
    return df

def add_cycles(df, aggregation = 'day'):
    timestamp_s = df.index.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    week = day * 7
    year = day * 365.2524

    df['year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    df['week_sin'] = np.sin(timestamp_s * (2 * np.pi / week))
    df['week_cos'] = np.cos(timestamp_s * (2 * np.pi / week))
    
    if aggregation == 'day':
        return df 
    else: 
        df['day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        df['day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        return df

def add_holidays(df):
    #Holidays Information
    years = range(2015,2022)
    holidays = []
    for year in years: 
        year_holidays = holidays_co.get_colombia_holidays_by_year(year)
        for day in year_holidays:
            holidays.append(day.date)

    holidays = pd.Series(pd.to_datetime(holidays))

    # Holidays
    time_ = df.index.normalize()
    sundays = pd.Series((df.index.weekday == 6).astype(int))
    df_holidays = time_.isin(holidays)
    final_holidays = sundays.mask(df_holidays, 1)
    return final_holidays.values

def temporal_variables(df, aggregation = 'day'):
    add_cycles(df, aggregation)
    df['holiday'] = add_holidays(df)
    df['saturday'] = pd.Series((df.index.weekday == 5).astype(int)).values
    return df

def aggreagtion_func(df, aggregation = '15-mins'):
    """
    Aggregates transactions by the given aggregation parameter. 
    
    Parameters:
    -----------
    - df: Pandas DataFrame, 
        Transactions by station. 
    - aggregation: str, default = '15-mins'. 
        Aggregation interval {'15-mins','hour','day','month'}
    """

    if aggregation == '15-mins': 
        hour = df.index.hour
        df = df[~hour.isin([0,1,2,3,23])]   
        return df 
    
    elif aggregation == 'hour':
        hours = df.resample('H').sum()
        hour = hours.index.hour
        return hours[~hour.isin([0,1,2,3,23])] 
    
    elif aggregation == 'day':
        hour = df.index.hour
        df = df[~hour.isin([0,1,2,3,23])]  
        return df.resample('D').sum()

    else:
        raise ValueError ('parameter {} not understood. Aggregation one of {15-mins,hour,day,month}'.format(aggregation))

def train_index(df, train_date):

    try:
        date = pd.Timestamp(train_date)
        idx = df.index.get_loc(date)

    except KeyError: 
        date = pd.Timestamp(train_date) + pd.DateOffset(hours=4)
        idx= df.index.get_loc(date)

    return idx


def train_test_data(settings):         
    aggregation = settings['aggregation']
    train_date = settings['train_date']

    df = read_data(settings)
    df = aggreagtion_func(df, aggregation = aggregation)
    df = temporal_variables(df, aggregation = aggregation)
    
    idx = train_index(df, train_date)

    train = df[:idx]
    test = df[idx:]
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
    train_copy = train.copy(deep = True)
    test_copy = test.copy(deep = True)

    assert train.name == test.name 
    name = train.name
    
    #Normalize train data:
    train_time_series = train[name]
    mean_train, std_train = train_time_series.mean(), train_time_series.std()
    normalize_train_time_series = (train_time_series - mean_train)/(std_train)

    #Normalize test data:
    test_time_series = test[name]
    normalize_test_time_series = (test_time_series - mean_train)/(std_train)

    train_copy[name] = normalize_train_time_series
    test_copy[name] = normalize_test_time_series

    return train_copy, test_copy, mean_train, std_train

def unnormalize_predict(normalized_prediction, train_mean, train_std):
    return (normalized_prediction * train_std) + train_mean

def min_max(train, test, label_columns=[]):
    """ Returns the min-max normalzation of the train and test data
    Parameters: 
    - train: array-like. Train dataset 
    - test: array-like. Test dataset
    - label_columns: Columns to be normalized. If No argument is passed, all columns are normalized

    Return:
    - train min-max normalization
    - test min-max normalization 
    - min value for train. np.array
    - max value for train. np.array
    """
    train_copy = train.copy(deep = True)
    test_copy = test.copy(deep = True)

    min_x = train[label_columns].min(axis = 0)
    max_x = train[label_columns].max(axis = 0)


    z_train = (train_copy[label_columns] - min_x) / (max_x - min_x)
    z_test = (test_copy[label_columns] - min_x) / (max_x - min_x)

    train_copy[label_columns] =  z_train
    test_copy[label_columns] =  z_test
    return train_copy, test_copy, np.array(min_x), np.array(max_x) 

def unnormalize_min_max(normalized_prediction, min_x, max_x):
    " Unnormlaize a normalize number or array"
    #TO DO: Add assert Shape (n,j) of normalized must be the same on min_x [n,] or max_x[n,] 
    return normalized_prediction * (max_x - min_x) + min_x

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
               data, label_columns=None, train_date='2018-08-01', 
               val_date=None, batch_size=32):
        self.batch_size = batch_size
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        idx = train_index(data, train_date) #data.index.get_loc(train_date)
        self.train_df = data.iloc[:idx]
        self.test_df = data.iloc[idx - input_width - shift + 1:]

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(data.columns)}

        # Work out the window parameters.
        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        def __repr__(self):
            return '\n'.join([
                    f'Total window size: {self.total_window_size}',
                    f'Input indices: {self.input_indices}',
                    f'Label indices: {self.label_indices}',
                    f'Label column name(s): {self.label_columns}'])

def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

WindowGenerator.split_window = split_window

def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=False, 
      batch_size= self.batch_size)

    ds = ds.map(self.split_window)

    return ds

WindowGenerator.make_dataset = make_dataset

@property
def train(self):
    return self.make_dataset(self.train_df)

# @property
# def val(self):
#   return self.make_dataset(self.val_df)

@property
def test(self):
    return self.make_dataset(self.test_df)

WindowGenerator.train = train
# WindowGenerator.val = val
WindowGenerator.test = test

# Unit Test
# settings = read_settings()
# train, test = train_test_data(settings)
# # print(test["(02000) cabecera autopista norte"])
# # print(train["(02000) cabecera autopista norte"])
# print(train)
# settings = read_settings()
# df = read_data(settings)
# print(df)
