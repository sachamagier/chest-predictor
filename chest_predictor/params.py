import os

MODEL_PATH = os.path.join(os.path.expanduser('~'), "code", "sachamagier", "chest_predictor", 'models')

LOCAL_REGISTRY_PATH = os.path.join(os.path.expanduser('~'), "code", "sachamagier", "chest_predictor", 'mlops', "training_outputs")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
LABEL_NAMES = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'No Finding']

DATA_URL = 'https://storage.googleapis.com/chest-predictor-data/resized_dataset.zip'
DATA_DIR =os.path.join(os.path.expanduser('~'), "code", "sachamagier", 'chest-predictor','raw_data')
DATA_FNAME = 'resized_dataset.zip'
DATA_PATH_LABEL = "../raw_data/encoded_labels.csv"



#https://storage.googleapis.com/chest-predictor-data/resized_dataset_test.zip if you want to test the code
