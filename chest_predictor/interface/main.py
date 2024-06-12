import numpy as np
#import pandas as pd
import tensorflow as tf

from pathlib import Path
#from colorama import Fore, Style
#from dateutil.parser import parse
import zipfile

from chest_predictor.params import *
from chest_predictor.ml_logic.data import downloading_data, creating_batch_dataset
# from chest_predictor.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from chest_predictor.ml_logic.preprocessor import load_and_preprocess_from_path_label
from chest_predictor.ml_logic.encoders import load_and_encode_labels
from chest_predictor.ml_logic.registry import load_model, save_model, save_results

#from chest_predictor.ml_logic.registry import load_model, save_model, save_results
#from chest_predictor.ml_logic.registry import mlflow_run, mlflow_transition_model



##### This should go in data.py #################################

AUTOTUNE = tf.data.AUTOTUNE

def creating_batch_dataset(dataset, BATCH_SIZE, AUTOTUNE):
    ds = dataset.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

############################################################


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

    load_data_to_bq(  ####This part still needs to be edited a bit so that it works with dataset
        image_label_ds,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'processed_{DATA_SIZE}',
        truncate=True
    )

    print("✅ preprocess() done \n")




############################## WORKING ON THE FOLLOWING ##############################
def train(
        batch_size = 32,
        patience = 5
    ) -> float:

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset
    - Store training results and model weights

    Return metrics: In this case auc...
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed data..." + Style.RESET_ALL)

    # Load processed data using `get_data_with_cache` in chronological order

    # Below, our columns are called ['_0', '_1'....'_66'] on BQ, student's column names may differ
    query = f"""
        SELECT * EXCEPT(_0)
        FROM `{GCP_PROJECT}`.{BQ_DATASET}.processed_{DATA_SIZE}
    """

    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{min_date}_{max_date}_{DATA_SIZE}.csv")
    data_processed = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=False
    )

    # if data_processed.shape[0] < 10:
    #     print("❌ Not enough processed data retrieved to train on")
    #     return None

    # Create (train_ds, val_ds and test_ds)
    #first the proportions of each set
    train_size = int(0.6 * len(all_image_paths))
    val_size = int(0.2 * len(all_image_paths))
    test_size = int(0.2 * len(all_image_paths))

    ###
    train_ds = image_label_ds.take(train_size)
    val_ds = image_label_ds.skip(train_size).take(val_size)
    test_ds = image_label_ds.skip(train_size + val_size).take(test_size)

    #load the data in batches
    ds = creating_batch_dataset(image_label_ds, batch_size, AUTOTUNE)
    train_ds = creating_batch_dataset(train_ds, batch_size, AUTOTUNE)
    val_ds = creating_batch_dataset(val_ds, batch_size, AUTOTUNE)
    test_ds = creating_batch_dataset(test_ds, batch_size, AUTOTUNE)
    # Train model using `model.py`
    model = load_model()

    if model is None:
        model = initialize_model(input_shape=(224, 224, 3))# ---> the initialize_model() function needs to be created in model.py

    model = compile_model(model, learning_rate=learning_rate)#--> same as compile_model
    model, history = train_model(
        model, train_ds,
        batch_size=batch_size,
        patience=patience,
        validation_data=val_ds
    )

    val_auc = np.min(history.history['val_mae']) # ---> if there is another metric change it here



    # params = dict(
    #     context="train",
    #     training_set_size=DATA_SIZE,
    #     row_count=len(X_train_processed),
    # )

    # Save results on the hard drive using taxifare.ml_logic.registry
    save_results(params=params, metrics=dict(mae=val_auc))

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    # # The latest model should be moved to staging
    # if MODEL_TARGET == 'mlflow':
    #     mlflow_transition_model(current_stage="None", new_stage="Staging")

    # print("✅ train() done \n")

    return val_auc


def evaluate(
        min_date:str = '2014-01-01',
        max_date:str = '2015-01-01',
        stage: str = "Production"
    ) -> float:
    """
    Evaluate the performance of the latest production model on processed data
    Return MAE as a float
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    model = load_model(stage=stage)
    assert model is not None

    min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    # Query your BigQuery processed table and get data_processed using `get_data_with_cache`
    query = f"""
        SELECT * EXCEPT(_0)
        FROM `{GCP_PROJECT}`.{BQ_DATASET}.processed_{DATA_SIZE}
        WHERE _0 BETWEEN '{min_date}' AND '{max_date}'
    """

    data_processed_cache_path = Path(f"{LOCAL_DATA_PATH}/processed/processed_{min_date}_{max_date}_{DATA_SIZE}.csv")
    data_processed = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=False
    )

    if data_processed.shape[0] == 0:
        print("❌ No data to evaluate on")
        return None

    data_processed = data_processed.to_numpy()

    X_new = data_processed[:, :-1]
    y_new = data_processed[:, -1]

    metrics_dict = evaluate_model(model=model, X=X_new, y=y_new)
    mae = metrics_dict["mae"]

    params = dict(
        context="evaluate", # Package behavior
        training_set_size=DATA_SIZE,
        row_count=len(X_new)
    )

    save_results(params=params, metrics=metrics_dict)

    print("✅ evaluate() done \n")

    return mae


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    if X_pred is None:
        X_pred = pd.DataFrame(dict(
        pickup_datetime=[pd.Timestamp("2013-07-06 17:18:00", tz='UTC')],
        pickup_longitude=[-73.950655],
        pickup_latitude=[40.783282],
        dropoff_longitude=[-73.984365],
        dropoff_latitude=[40.769802],
        passenger_count=[1],
    ))

    model = load_model()
    assert model is not None

    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    return y_pred
