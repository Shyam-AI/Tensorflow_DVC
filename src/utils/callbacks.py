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
    checkpoint_file_path = os.path.join(checkpoint_dir, "ckpt_model.h5")
    checkpoint_callbacks = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file_path, save_best_only= True)

    ckpt_callbacks_filepath = os.path.join(call_back_dir, "checkpoint_cb.cb")
    joblib.dump(checkpoint_callbacks, ckpt_callbacks_filepath)

    logging.info(f"Checkpoints callback is being saved at {ckpt_callbacks_filepath}")


def get_callbacks(callback_dir_path):
    callback_path = [os.path.join(callback_dir_path, bin_file) for bin_file in os.listdir(callback_dir_path)
    if bin_file.endswith(".cb")]

    call_backs = [joblib.load(path) for path in callback_path]
    logging.info(f"saved call_bakcs are loaded from {callback_dir_path}")
    return call_backs