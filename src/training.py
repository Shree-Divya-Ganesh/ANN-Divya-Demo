import sys
sys.path.append("E:\Demo-DLCVNLP\ANN-Divya-Demo")

from src.utils.common import read_config
from src.utils.data_prep import get_data
from src.utils.model import create_model
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

    history = model.fit(x_train, y_train , epochs = NO_OF_EPOCHS, validation_data = VALIDATION_SET)

if __name__ == "__main__":
    
    args = argparse.ArgumentParser()

    args.add_argument("--config","-c", default="config.yaml")

    parsed_args = args.parse_args()

    training(config_path = parsed_args.config)