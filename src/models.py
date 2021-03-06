from enum import unique
import tensorflow as tf
import os

from tensorflow.python.keras.layers.advanced_activations import Softmax
from tensorflow.python.keras.layers.core import Flatten

import joblib
import logging

from src.utils.all_utils import get_timestamp


def get_VGG_16_model(input_shape, model_path):
    model = tf.keras.applications.vgg16.VGG16(input_shape=input_shape, weights="imagenet", include_top=False)
    model.save(model_path)
    logging.info(f"VGG16 saved at {model_path}")
    return model


def prepare_model(model, CLASSES, freeze_all, freeze_till, learning_rate):
    if freeze_all:
        for layer in model.layers:
            layer.trainable = False
    elif freeze_till is not None and freeze_till > 0:
        for layer in model.layers[:freeze_till]:
            layer.trainable = False

    # add fully-connected layers
    flatten_in = tf.keras.layers.Flatten()(model.output)
    prediction = tf.keras.layers.Dense(units = CLASSES,activation="softmax" )(flatten_in)

    full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)

    full_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    logging.info(f"custom model is compiled and ready to be trained")
    # full_model.summary()

    return full_model


def load_full_model(untrained_full_model_path):
    model = tf.keras.models.load_model(untrained_full_model_path)
    logging.info(f"untrained model loaded from {untrained_full_model_path}")
    return model


def get_unique_path_to_save_model(trained_model_dir, default_model_name="Model"):
    timestamp = get_timestamp(default_model_name)
    unique_model_name = f"{timestamp}.h5"
    unique_model_path = os.path.join(trained_model_dir, unique_model_name)

    return unique_model_path