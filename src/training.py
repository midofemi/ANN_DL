"""
Training.py is essential. This file will be pointing to all the other functions
"""

import os
from utils.common import read_config
from utils.data_mgmt import get_data
from utils.model import create_model, save_model, save_plot
from utils.callbacks import get_callbacks
import argparse

""""
I'm not sure but this same framework can be used to write modular code for any modeling problem
"""

def training(config_path):
    """
    Read our config paramter. It a good idea to pass it (config) in the training function because this is where we will be training our model
    Thus many parameter need to be called in this function
    """
    config = read_config(config_path)
    
    validation_datasize = config["params"]["validation_datasize"] # Here we just saying. inside param, get the validation_datasize and pass it to get_data function
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize) 

    #Initiate yaml file which will be passed on our create model function
    LOSS_FUNCTION = config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NUM_CLASSES = config["params"]["num_classes"]

    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)

    EPOCHS = config["params"]["epochs"]
    VALIDATION_SET = (X_valid, y_valid)

    # # create callbacks
    # create callbacks
    CALLBACK_LIST = get_callbacks(config, X_train)

    history = model.fit(X_train, y_train, epochs=EPOCHS,
                    validation_data=VALIDATION_SET, callbacks=CALLBACK_LIST)

    # _____________________________________________________________________________________________________________
    """
    #Note: The artifacts\model folder was not created by me. It was automatically created when I ran the script and also
    #saved the model in an h5 format
    """
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]
    plots_dir = config["artifacts"]["plots_dir"]
    plot_name = config["artifacts"]["plot_name"]
    
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    # _____________________________________________________________________________________________________________
    
    model_name = config["artifacts"]["model_name"]


    plot_dir_path = os.path.join(artifacts_dir, plots_dir)
    os.makedirs(plot_dir_path, exist_ok=True)

    #Save Model in H5: H5 is just how we save deep learning model
    save_model(model, model_name, model_dir_path)

    save_plot(history, plot_name, plot_dir_path)

if __name__ == '__main__':
    #################### THIS JUST HELPS US TO USED THOSE PARAMETERS IN THE YAML FILE #################################
    #First thing to do if you're using a yaml file in a function
    args = argparse.ArgumentParser() #This comes with a library called argparse
    """
    This can be passed on terminal also. Let say we have another config file called config2 and we want to pass that config file in the terminal rather than config.yaml
    We can say: python src/training.py --config = config2.yaml. This will execute those parameter in config2 instead of config
    """
    args.add_argument("--config", "-c", default="config.yaml") 

    parsed_args = args.parse_args()
    ####################################################################################################################

    training(config_path=parsed_args.config)