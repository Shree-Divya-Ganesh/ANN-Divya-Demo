import sys
sys.path.append("E:/Demo-DLCVNLP/ANN-Divya-Demo")

from src.utils.common import read_config
from src.utils.data_prep import get_data
from src.utils.model import create_model, save_model, save_plot
from src.utils.callbacks import get_callbacks
import os
import argparse

def training(config_path):
    config = read_config(config_path)
    print(config)
    validation_datasize = config['params']['validation_datasize']
    (x_train, y_train), (x_valid, y_valid), (x_test,y_test)  = get_data(validation_datasize)

    LOSS_FUNCTION = config['params']['loss_function']
    OPTIMIZER = config['params']['optimizer']
    METRICS = config['params']['metrics']
    NO_CLASSES = config['params']['no_classes']

    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NO_CLASSES)

    NO_OF_EPOCHS = config['params']['epochs']
    VALIDATION_SET = (x_valid, y_valid)


    CALLBACK_LIST = get_callbacks(config, x_train)

    history = model.fit(x_train, y_train , epochs = NO_OF_EPOCHS, validation_data = VALIDATION_SET, callbacks =[CALLBACK_LIST])

    model_name = config['artifacts']['model_name']
    model_dir = config['artifacts']['model_dir']
    artifacts_dir = config['artifacts']['artifacts_dir']

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok = True)

    save_model(model, model_name, model_dir_path)

    # plots_dir = config['artifacts']['plots_dir']
    save_plot_dir= config['artifacts']['plots_dir']
    save_plot_dir_path = os.path.join(artifacts_dir, save_plot_dir)
    os.makedirs(save_plot_dir_path, exist_ok = True)

    plot_name = config['artifacts']['plot_name']
    save_plot(history, plot_name, save_plot_dir_path)

if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c", default="config.yaml")
    parsed_args = args.parse_args()
    training(config_path = parsed_args.config)