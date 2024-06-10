import pandas as pd

from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path
import tensorflow as tf
import os
import shutil

tf.__version__

project_name = "chest-predictor"
username = os.environ.get('USER')

#Set variables (origin url, destination folder, folder name)
data_url = 'https://storage.googleapis.com/chest-predictor-data/resized_dataset.zip'
data_dir = f"/Users/{username}/code/sachamagier/{project_name}/raw_data"
data_fname = 'resized_dataset.zip'

#Creating a list of the 15 different labels
label_names = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'No Finding']



def downloading_data(data_dir,data_fname,data_url):
    data_root_orig = tf.keras.utils.get_file(
    fname=os.path.join(data_dir, data_fname),
    origin=data_url
)

# #This remove the '__MACOSX' file that is created on Mac Laptops
# if Path(os.path.join(data_dir, "__MACOSX")).is_dir():
#     # remove the __MACOSX folder if it exists
#     shutil.rmtree("../raw_data/__MACOSX")

#Get all image paths
    all_image_paths = [str(path) for path in (data_root/"images"/"set_full").iterdir()]

# it will find all (256,256,4) images and delete them
# from PIL import Image

# ## specify your dataset directory
# data_directory = data_root/"images"/"set_full"

# ## list to store the names of the images to be removed
# images_to_remove = []

# ## iterate over all files in the dataset directory
# for filename in os.listdir(data_directory):
#     if filename.endswith(".png"):  # make sure it's a png file
#         file_path = os.path.join(data_directory, filename)
#         image = Image.open(file_path)  # load the image with Pillow

#         ## convert the image to a numpy array to check its shape
#         #image_array = np.array(image)

#         ## check the shape of the image
#         #if image_array.shape == (256, 256, 4):
#             #images_to_remove.append(filename)  # add the filename to the list if it's to be removed

# ## uncomment the following line if you're sure about the images to be removed
# # [os.remove(os.path.join(data_directory, filename)) for filename in images_to_remove]
    return all_image_paths


def load_labels(data_root_orig,all_images_paths):
    labels_df = pd.read_csv(data_root_orig /"Data_Entry_2017.csv")
    labels_df.set_index('Image Index', inplace=True)

    # Create list of all image labels
    all_image_labels = labels_df.loc[[os.path.basename(path) for path in all_image_paths], 'Finding Labels'].values

    return all_image_labels

def encoding_labels(label_names, all_image_labels):
    # Vectorized encoding of labels
    def encode_labels(labels):
        split_labels = tf.strings.split(labels, '|')
        encoded_labels = string_lookup(split_labels)
        one_hot_encoded = tf.reduce_sum(tf.one_hot(encoded_labels, depth=len(label_names)), axis=0)
        return one_hot_encoded
    ## Apply encoding to all labels in a vectorized manner
    encoded_values = tf.stack([encode_labels(labels) for labels in all_image_labels])
    #Converting to numpy array for printing
    encoded_values_np = encoded_values.numpy()
    # Convert encoded values to tensor slices for the dataset
    label_ds = tf.data.Dataset.from_tensor_slices(encoded_values)

    return label_ds
