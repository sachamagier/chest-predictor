import pandas as pd

from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path
import tensorflow as tf

tf.__version__

project_name = "chest-predictor"
username = os.environ.get('USER')

#Set variables (origin url, destination folder, folder name)
data_url = 'https://storage.googleapis.com/chest-predictor-data/resized_dataset.zip'
data_dir = f"/Users/{username}/code/sachamagier/{project_name}/raw_data"
data_fname = 'resized_dataset.zip'


def downloading_data(data_dir,data_fname,data_url):
    data_root_orig = tf.keras.utils.get_file(
    fname=os.path.join(data_dir, data_fname),
    origin=data_url
)
    return data_root_orig


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
