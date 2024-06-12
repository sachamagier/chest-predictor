import pandas as pd
from chest_predictor.params import *
import tensorflow as tf
import os
from pathlib import Path
import zipfile
import shutil
from PIL import Image
import numpy as np





tf.__version__


def downloading_data(DATA_DIR,DATA_FNAME,DATA_URL):
    data_root = tf.keras.utils.get_file(
    fname=os.path.join(DATA_DIR, DATA_FNAME),
    origin=DATA_URL
)


#Extract the dataset
    with zipfile.ZipFile(data_root, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)

    #Set the new data_root
    data_root = Path(os.path.join(DATA_DIR, 'resized_dataset'))

# # #This remove the '__MACOSX' file that is created on Mac Laptops
    if Path(os.path.join(DATA_DIR, "__MACOSX")).is_dir():
#         # remove the __MACOSX folder if it exists
        shutil.rmtree(os.path.join(DATA_DIR, "__MACOSX"))
    print(f"✅ Data saved in your computer")

#Get all image paths
    data_root = Path(data_root)
    all_image_paths = [str(path) for path in (data_root/"images"/"set_full").iterdir()]
    print("Get all image paths:")
    print(len(all_image_paths))

    return all_image_paths


def data_clean(DATA_DIR):
    # List to store the names of the images to be removed
    images_to_remove = []

    # Iterate over all files in the dataset directory
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".png"):  # make sure it's a png file
            file_path = os.path.join(DATA_DIR, filename)
            image = Image.open(file_path)  # load the image with Pillow

            # Convert the image to a numpy array to check its shape
            image_array = np.array(image)

            # Check the shape of the image
            if image_array.shape == (256, 256, 4):
                images_to_remove.append(filename)  # add the filename to the list if it's to be removed

    # Print the names of the images to be removed
    # print("Images to be removed: ", images_to_remove)
    ## uncomment the following line if you're sure about the images to be remove
    # # Remove the images from the directory
    for filename in images_to_remove:
        file_path = os.path.join(os.path.join(DATA_DIR, "resized_dataset", "images", "set_full"), filename)
        os.remove(file_path)
    print(f"✅ Data removed in your computer")




# def load_and_encode_labels(DATA_PATH_LABEL):
#     """
#     Load encoded labels from a CSV file. and then transform them into a tensor

#     Args:
#     - data_path (str): Path to the CSV file containing encoded labels
#     """
#     # Read the CSV file
#     df_labels = pd.read_csv(DATA_PATH_LABEL)

#     # Convert the labels into a TensorFlow tensor
#     encoded_values = tf.convert_to_tensor(df_labels.values, dtype=tf.float32)
#     label_ds = tf.data.Dataset.from_tensor_slices(encoded_values)
#     # Depending on if we are doing this in another .py or not we can add the nextg line. It creates a Dataset with image paths and encoded labels
#     #path_label_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, encoded_values)) --> add all_image_paths to the function parameters
#     return label_ds



if __name__ == '__main__':
    downloading_data(DATA_DIR,DATA_FNAME,DATA_URL)
    data_clean(os.path.join(DATA_DIR, "resized_dataset", "images", "set_full"))


    # Load and encode the labels
    # label_ds = load_and_encode_labels(DATA_PATH_LABEL)
