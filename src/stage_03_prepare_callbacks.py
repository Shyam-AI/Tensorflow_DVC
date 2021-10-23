from src.utils.all_utils import read_yaml, create_directory
from src.models import get_VGG_16_model, prepare_model
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