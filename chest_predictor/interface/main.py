import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from chest_predictor.params import *
from chest_predictor.ml_logic.data import downloading_data
# from chest_predictor.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from chest_predictor.ml_logic.preprocessor import *
from chest_predictor.ml_logic.registry import load_model, save_model, save_results
#from chest_predictor.ml_logic.registry import mlflow_run, mlflow_transition_model


data_dir = f"/Users/{username}/code/sachamagier/{project_name}/raw_data"


def preprocess(min_date:str = '2009-01-01', max_date:str = '2015-01-01') -> None:
    """
    """
    # Check if the data has been downloaded
    data_fname = 'resized_dataset.zip'
    data_url = 'https://storage.googleapis.com/chest-predictor-data/resized_dataset.zip'

    data_root_orig = Path(os.path.join(data_dir, 'resized_dataset'))
    if not os.path.exists(data_root_orig):
        print("Data is not downloaded. Downloading now...")
        downloading_data(data_dir, data_fname, data_url)
    else:
        print("Data is already downloaded.")
    # Retrieve data using `get_data`
    '''this needs to be changed'''
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f".csv")
    data_query = get_data_with_cache(
        query=query,
        gcp_project=GCP_PROJECT,
        cache_path=data_query_cache_path,
        data_has_header=True
    )
    #import zipfile

    #Extract the dataset
    #with zipfile.ZipFile(data_root_orig, 'r') as zip_ref:
        #zip_ref.extractall(data_dir)

    #set the new data_root
    data_root = Path(os.path.join(data_dir, 'resized_dataset'))
    #Get all image paths
    all_image_paths = [str(path) for path in (data_root/"images"/"set_full").iterdir()]
    #from PIL import Image

    ## specify your dataset directory
    #data_directory = data_root/"images"/"set_full"

    ## list to store the names of the images to be removed
    #images_to_remove = []

    ## iterate over all files in the dataset directory
    #for filename in os.listdir(data_directory):
        #if filename.endswith(".png"):  # make sure it's a png file
            #file_path = os.path.join(data_directory, filename)
            #image = Image.open(file_path)  # load the image with Pillow

            ## convert the image to a numpy array to check its shape
            #image_array = np.array(image)

            ## check the shape of the image
            #if image_array.shape == (256, 256, 4):
                #images_to_remove.append(filename)  # add the filename to the list if it's to be removed

    ## print the names of the images to be removed
    #print("Images to be removed: ", images_to_remove)

    ## uncomment the following line if you're sure about the images to be removed
    # [os.remove(os.path.join(data_directory, filename)) for filename in images_to_remove]

    # Process data
    data_clean = clean_data(data_query)

    X = data_clean.drop("fare_amount", axis=1)
    y = data_clean[["fare_amount"]]

    X_processed = preprocess_features(X)

    # Load a DataFrame onto BigQuery containing [pickup_datetime, X_processed, y]
    # using data.load_data_to_bq()
    data_processed_with_timestamp = pd.DataFrame(np.concatenate((
        data_clean[["pickup_datetime"]],
        X_processed,
        y,
    ), axis=1))

    load_data_to_bq(
        data_processed_with_timestamp,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'processed_{DATA_SIZE}',
        truncate=True
    )

    print("✅ preprocess() done \n")
