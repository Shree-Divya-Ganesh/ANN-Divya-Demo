import tensorflow as tf
# from tf.keras.callbacks import EarlyStopping
import numpy as np
import os
import time
import yaml

def get_timestamp(name):
    timestamp = time.asctime().replace(' ', '_').replace(':','_')
    unique_name = f"{name}_at_{timestamp}"
    return unique_name

def get_callbacks(config, X_train):
    logs = config['logs']
    uniq_dir_name = get_timestamp("tb_logs")
    TENSORBOARD_ROOT_LOG_DIR = os.path.join(logs['logs_dir'], logs['TENSORBOARD_ROOT_LOG_DIR'], uniq_dir_name)

    os.makedirs(TENSORBOARD_ROOT_LOG_DIR, exist_ok = True)

    tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir = TENSORBOARD_ROOT_LOG_DIR)
    fileWriter = tf.summary.create_file_writer(logdir = TENSORBOARD_ROOT_LOG_DIR)

    with fileWriter.as_default():
        images = np.reshape(X_train[10:30],(-1,28,28,1))
        tf.summary.image("20 Hand written digit samples", images, max_outputs = 25, step=0)

    params = config['params']
    earlystopping = tf.keras.callbacks.EarlyStopping(
        patience = params['patience'], 
        restore_best_weights = params['restore_best_weights'])

    artifacts = config['artifacts']
    CKPT_DIR = os.path.join(artifacts["artifacts_dir"], artifacts['checkpoint_dir'])
    os.makedirs(CKPT_DIR, exist_ok = True)

    checkpoints_path = os.path.join(CKPT_DIR, "model_ckpt.h5")

    checkpointing = tf.keras.callbacks.ModelCheckpoint(checkpoints_path, save_best_only = True)

    callback_list = [tensorboard_callbacks, earlystopping, checkpointing]

    return [callback_list]

