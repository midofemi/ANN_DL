import os
from utils.common import read_config
from utils.data_mgmt import get_data
#from src.utils.model import create_model, save_model
#from src.utils.callbacks import get_callbacks
import argparse

def training(config_path):
    #Read our config paramter
    config = read_config(config_path)
    
    validation_datasize = config["params"]["validation_datasize"] # Here we just saying. inside param, get the validation_datasize and pass it to get_data function
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize) 
    # LOSS_FUNCTION = config["params"]["loss_function"]
    # OPTIMIZER = config["params"]["optimizer"]
    # METRICS = config["params"]["metrics"]
    # NUM_CLASSES = config["params"]["num_classes"]

    # model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)

    # EPOCHS = config["params"]["epochs"]
    # VALIDATION_SET = (X_valid, y_valid)

    # # create callbacks
    # CALLBACK_LIST = get_callbacks(config, X_train)

    # history = model.fit(X_train, y_train, epochs=EPOCHS,
    #                 validation_data=VALIDATION_SET, callbacks=CALLBACK_LIST)

    # artifacts_dir = config["artifacts"]["artifacts_dir"]
    # model_dir = config["artifacts"]["model_dir"]
    
    # model_dir_path = os.path.join(artifacts_dir, model_dir)
    # os.makedirs(model_dir_path, exist_ok=True)
    
    # model_name = config["artifacts"]["model_name"]

    # save_model(model, model_name, model_dir_path)

if __name__ == '__main__':
    #################### THIS JUST HELPS US TO USED THOSE PARAMETERS IN THE YAML FILE #################################
    #First thing to do if you're using a yaml file in a function
    args = argparse.ArgumentParser() #This comes with a library called argparse

    args.add_argument("--config", "-c", default="config.yaml")

    parsed_args = args.parse_args()
    ####################################################################################################################

    training(config_path=parsed_args.config)