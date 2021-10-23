from src.utils.all_utils import read_yaml, create_directory
from src.utils.callbacks import create_and_save_tensorboard_callbacks, create_and_save_checkpoint_callbacks
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


def prepare_callbacks(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]

    tensorboard_root_log_dir = artifacts["TENSORBOARD_ROOT_LOG_DIR"]
    tensorboard_log_dir = os.path.join(artifacts_dir, tensorboard_root_log_dir)

    checkpoint_dir = os.path.join(artifacts_dir, artifacts["CHECKPOINTS_DIR"])

    call_back_dir = os.path.join(artifacts_dir, artifacts["CALLBACKS_DIR"])

    #Create these directories
    create_directory([tensorboard_log_dir, checkpoint_dir, call_back_dir])

    create_and_save_tensorboard_callbacks(call_back_dir, tensorboard_log_dir)
    create_and_save_checkpoint_callbacks(call_back_dir, checkpoint_dir)





if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="param.yaml")
 
    parsed_args = args.parse_args()

    try:
        logging.info(">>>>> stage three started")
        prepare_callbacks(config_path=parsed_args.config, params_path= parsed_args.params)
        logging.info("stage three started completed! callbacks saved as binary >>>>>\n")
    except Exception as e:
        logging.exception(e)
        raise e