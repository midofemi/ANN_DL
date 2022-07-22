"""
This file load our config.yaml file and read it so it can used
"""
import yaml

def read_config(config_path):
    with open(config_path) as config_file:
        content = yaml.safe_load(config_file)
    return content