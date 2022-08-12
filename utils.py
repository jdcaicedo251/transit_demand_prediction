import os
import json 
import joblib 
import yaml
import tensorflow as tf

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def save_json(path, dictionary):
    a_file = open(path, "w")
    json.dump(dictionary, a_file)
    a_file.close()

def read_json(path):
    with open(path) as f:
        return json.load(f)


def delete_files_folder(directory):
    logging.info('Overwrite True. Deleting files in {}'.format(directory)) 
    for f in os.listdir(directory):
        os.remove(os.path.join(directory, f))

def read_plk(directory):
    fpath = os.path.join(directory)
    plk = joblib.load(fpath)
    return plk

def read_yaml(path):
    with open(path, "r") as stream:
        try:
            yaml_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return yaml_file


def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'loss',
                                                    patience=patience,
                                                    mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=100,
                    #   validation_data=window.val,
                      callbacks=[early_stopping],
                      verbose = False)
    return history
