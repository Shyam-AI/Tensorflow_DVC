import tensorflow as tf
import os
import joblib
import logging
from src.utils.all_utils import get_timestamp




def create_and_save_tensorboard_callbacks(call_back_dir, tensorboard_log_dir):
    unique_name = get_timestamp("tb_logs")

    tb_running_logs_dir = os.path.join(tensorboard_log_dir, unique_name)
    tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir= tb_running_logs_dir)

    tb_callbacks_filepath = os.path.join(call_back_dir, "tensorboard_cb.cb")
    joblib.dump(tensorboard_callbacks, tb_callbacks_filepath)

    logging.info(f"Tensorboard callback is being saved at {tb_callbacks_filepath}")


def create_and_save_checkpoint_callbacks(call_back_dir, checkpoint_dir):
    pass