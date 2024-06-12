import numpy as np
#import pandas as pd
import tensorflow as tf

from pathlib import Path
#from colorama import Fore, Style
#from dateutil.parser import parse
import zipfile

from chest_predictor.params import *
from chest_predictor.ml_logic.data import downloading_data
# from chest_predictor.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from chest_predictor.ml_logic.preprocessor import load_and_preprocess_from_path_label
from chest_predictor.ml_logic.encoders import load_and_encode_labels

#from chest_predictor.ml_logic.registry import load_model, save_model, save_results
#from chest_predictor.ml_logic.registry import mlflow_run, mlflow_transition_model


data_dir = f"/Users/{username}/code/sachamagier/{project_name}/raw_data"
data_fname = 'resized_dataset.zip'


def preprocess() -> None:
    """
    - Download data if not already downloaded
    - Extract the dataset if not already extracted
    - Get all image paths (and remove invalid images--> to be implemented)
    """
    # Check if the data has been downloaded
    data_root_orig = Path(os.path.join(data_dir, 'resized_dataset'))
    if not os.path.exists(data_root_orig):
        print("Data is not downloaded. Downloading  an unzipping...")
        data_root_orig = downloading_data(data_dir, data_fname, data_url) ########### make these global vars
        with zipfile.ZipFile(data_root_orig, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    elif not (data_root_orig/"images"/"set_full").exists():
        print("Unzipping file")
        # Extract the dataset
        with zipfile.ZipFile(data_root_orig, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    else:
        print('Data ready for use...')

    # Retrieve data using `get_data`
    all_image_paths = [str(path) for path in (data_root / "images" / "set_full").iterdir()]
    encoded_values = load_and_encode_labels(data_path)

    # Process data
    #Creates a new dataset that loads and formats images on the fly by mapping
    # preprocess_image over the dataset of paths.
    path_label_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, encoded_values))
    image_label_ds = path_label_ds.map(load_and_preprocess_from_path_label)

    load_data_to_bq(
        image_label_ds,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'processed_{DATA_SIZE}',
        truncate=True
    )

    print("✅ preprocess() done \n")
