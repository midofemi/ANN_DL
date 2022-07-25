import tensorflow as tf
import os
import numpy as np
import time

def get_timestamp(name):
    #This gives you the time when the project is run which our folder name will be created
    timestamp = time.asctime().replace(" ", "_").replace(":", "_")
    unique_name = f"{name}_at_{timestamp}"
    return unique_name

def get_callbacks(config, X_train):
    logs = config["logs"]
    unique_dir_name = get_timestamp("tb_logs") #This will create the folder tb_logs at a unique time stamps from get_timestamp
    TENSORBOARD_ROOT_LOG_DIR = os.path.join(logs["logs_dir"], logs["TENSORBOARD_ROOT_LOG_DIR"], unique_dir_name)

    os.makedirs(TENSORBOARD_ROOT_LOG_DIR, exist_ok=True)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_ROOT_LOG_DIR)
    #_________________________________________________________________________________
    """
    This write the logs into tensorboard itself (File_Writer)
    """
    file_writer = tf.summary.create_file_writer(logdir=TENSORBOARD_ROOT_LOG_DIR)
    #___________________________________________________________________________________
    """
    This write the images into tensorboard itself (File_Writer)
    """
    with file_writer.as_default():
        """
        Take 20 images from x_train that what X_train[10:30] means.
        And inside those 20 imgs, each img has a 28by28 dimension
        We did (-1,28,28,1) because we want to keep those 20 images in batches (1batch of 20 images)
        """
        images = np.reshape(X_train[10:30], (-1, 28, 28, 1)) ### <<< 20, 28, 28, 1
        tf.summary.image("20 handritten digit samples", images, max_outputs=25, step=0)
    #_______________________________________________________________________________________
    params = config["params"]
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=params["patience"], 
        restore_best_weights=params["restore_best_weights"])

    artifacts = config["artifacts"]
    CKPT_dir = os.path.join(artifacts["artifacts_dir"], artifacts["CHECKPOINT_DIR"])
    os.makedirs(CKPT_dir, exist_ok=True)

    CKPT_path = os.path.join(CKPT_dir, "model_ckpt.h5")

    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path, save_best_only=True)

    return [tensorboard_cb, early_stopping_cb, checkpointing_cb]