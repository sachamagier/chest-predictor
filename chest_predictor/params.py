import os

LOCAL_REGISTRY_PATH = os.path.join(os.path.expanduser('~'), "code", "sachamagier", "chest_predictor", mlops, "training_outputs")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
