import os
import json 
import joblib 
import yaml

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


