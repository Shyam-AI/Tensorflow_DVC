from src.utils.all_utils import read_yaml, create_directory
from src.models import load_full_model, get_unique_path_to_save_model
from src.utils.callbacks import get_callbacks
from src.utils.data_management import train_valid_generator
import argparse
import pandas as pd
import os
from tqdm import tqdm     #prints the progress bar when we run the for loop
import logging
import io


logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'running_logs.log'), level=logging.INFO, format=logging_str,
                    filemode="a")


def train_model(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]

    train_model_dir_path = os.path.join(artifacts_dir, artifacts["TRAINED_MODEL_DIR"])
    create_directory([train_model_dir_path])

    untrained_full_model_path = os.path.join(artifacts_dir, artifacts["BASE_MODEL_DIR"], artifacts["UPDATED_BASE_MODEL"])

    model = load_full_model(untrained_full_model_path)

    callback_dir_path = os.path.join(artifacts_dir, artifacts["CALLBACKS_DIR"] )
    callbacks = get_callbacks(callback_dir_path)

    train_generator, valid_generator = train_valid_generator(data_dir = artifacts["DATA_DIR"], IMAGE_SIZE= tuple(params["IMAGE_SIZE"][:-1]), BATCH_SIZE= params["BATCH_SIZE"], do_data_augmentation= params["AUGMENTATION"])

    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = valid_generator.samples // valid_generator.batch_size

    model.fit(
        train_generator,
        validation_data = valid_generator,
        epochs = params["EPOCHS"],
        steps_per_epoch = steps_per_epoch,
        validation_steps = validation_steps,
        callbacks = callbacks

    )
    logging.info(f"Training done!!!!")
    trained_model_dir = artifacts["TRAINED_MODEL_DIR"]
    create_directory([trained_model_dir])

    model_path = get_unique_path_to_save_model(trained_model_dir)

    model.save(model_path)

    logging.info(f"Trained model is saved at the location {model_path}")




if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="param.yaml")
 
    parsed_args = args.parse_args()

    try:
        logging.info(">>>>> stage four started")
        train_model(config_path=parsed_args.config, params_path= parsed_args.params)
        logging.info("stage four started completed! Training completed and model is saved >>>>>\n")
    except Exception as e:
        logging.exception(e)
        raise e