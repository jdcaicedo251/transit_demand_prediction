import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import os

import data 
from utils import read_json 

INPUT_PATH = 'data/'
METRICS = ['mape', 'vape']

def read_target(df, station, aggregation):
    data = df[station]

    if aggregation == 'day':
        forecast_window = 7
    elif aggregation == 'hour':
        forecast_window = 24
    elif aggregation == 'month':
        forecast_window = 12
    elif aggregation == '15mins':
        forecast_window = 8
    else:
        raise KeyError ('aggregation parameter is one of [day, hour, month, 15 mins]')

    shift_list = []
    col_list = []
    for i in range(forecast_window):
        shift_list.append(data.shift(-i))
        col_list.append('forecast_period_' + str(i + 1))
    target = pd.concat(shift_list, axis = 1)
    target.columns = col_list
    target = np.array(target.dropna())
    assert target.shape[1] == forecast_window
    return target

def read_prediction(model, aggregation, station):
    path = os.path.join('output', model, 'results', aggregation , station + '.json' )
    prediction = read_json(path)
    prediction = np.array(prediction['prediction'])
    return np.expm1(prediction)

def APE(target, predicted):
    return np.abs((target - predicted) / target)

def mape(target, predicted, axis = None):
    return np.mean(APE(target, predicted), axis = axis)

def maape(target, predicted, axis = None):
    ape = APE(target, predicted)
    return np.mean(np.arctan(ape), axis = axis)

def vape(target, predicted, axis = None):
    return np.var(APE(target, predicted), axis = axis)

def plot_before_after(pre, post, metric_name):

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize = (10,10))
    axs.scatter(pre, post, color="g")
    axs.plot([0, 1], [0, 1])
    axs.set_ylabel('Post COVID-19 \n [Jun 2019 - Feb 2020]')
    axs.set_xlabel('Pre COVID-19 \n [Jun 2019 - Feb 2020]')
    axs.set_title(metric_name, fontsize = 50)
    return None


def plot_evolution(metrics, metric_name):
    plt.plot(x = 'Date', y = 'MAPE')
    plt.ylabel(metric_name)
    plt.title('Evolution of ' + metric_name)
    plt.axvline(578, color = 'r')
    return None

def run_regression(pre, post, metric_name):

    for metric in METRICS:
        pre_name = 'pre_' + metric
        post_name = 'post_' + metric
        x = sm.add_constant(results[pre_name].rename('pre-COVID 19 metric'), prepend=False)

    return regression

## FIX ME 
def linear_regression(results, metrics):

    models_list = []

    for metric in metrics:
        pre_name = 'pre_' + metric
        post_name = 'post_' + metric
        x = sm.add_constant(results[pre_name].rename('pre-COVID 19 metric'), prepend=False)
        model = sm.OLS(results[post_name], x).fit()
        models_list.append(model)

    ols_results = summary_col(models_list,stars=True, info_dict = {"N":lambda x:(x.nobs)},
                              model_names = metrics, float_format='%.3f')

    return ols_results


def results(model, aggregation, metric):
    #read data
    train, test = data.split_data(INPUT_PATH, train_date = (2018, 8, 1), aggreagation = aggregation)
    stations = set(train.columns[train.columns.str.contains("\(")])

    #For each station
    w_metric = []
    s_pre_metric = []
    s_post_metric = []
    for station in stations:
        target = read_target(test, station, aggregation)
        prediction = read_prediction(station)
        ## FIX ME: Check 572 is the actual split of the data
        pre_target = target[:578,:]
        pre_prediction = prediction[:578,:]
        post_target = target[578:,:]
        post_prediction = prediction[578:,:]

        # METRICS
        window_metric = metric(target, prediction, axis = 1)
        pre_station_metric = metric(pre_target, pre_prediction)
        post_station_metric = metric(post_target, post_prediction)
        w_metric.append(window_metric)
        s_pre_metric.append(pre_station_metric)
        s_post_metric.append(post_station_metric)
    
    plot_before_after(s_pre_metric, s_post_metric)
    plot_evolution(w_metric)
    regression = run_regression(s_pre_metric, s_post_metric)

    return regression

## -------------------------- tests ------------------------------------
model = 'arima'
aggregation = 'day'
metric = 'mape'
station = "(02000) cabecera autopista norte"
train, test = data.split_data(INPUT_PATH, train_date = (2018, 8, 1), aggreagation = aggregation)

target = read_target(test, station, aggregation = aggregation)[:-1,:]
print(target.shape)
prediction = read_prediction(model, aggregation, station)[:target.shape[0],:]
print(prediction)

# m = maape(target, prediction)
# print (m)