import pandas as pd
from chest_predictor.params import *
import tensorflow as tf
import os

tf.__version__


def downloading_data(DATA_DIR,DATA_FNAME,DATA_URL):
    data_root = tf.keras.utils.get_file(
    fname=os.path.join(DATA_DIR, DATA_FNAME),
    origin=DATA_URL
)

# #This remove the '__MACOSX' file that is created on Mac Laptops
# if Path(os.path.join(DATA_DIR, "__MACOSX")).is_dir():
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




def load_and_encode_labels(DATA_PATH):
    """
    Load encoded labels from a CSV file. and then transform them into a tensor

    Args:
    - data_path (str): Path to the CSV file containing encoded labels
    """
    # Read the CSV file
    df_labels = pd.read_csv(DATA_PATH)

    # Convert the labels into a TensorFlow tensor
    encoded_values = tf.convert_to_tensor(df_labels.values, dtype=tf.float32)
    label_ds = tf.data.Dataset.from_tensor_slices(encoded_values)
    # Depending on if we are doing this in another .py or not we can add the nextg line. It creates a Dataset with image paths and encoded labels
    #path_label_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, encoded_values)) --> add all_image_paths to the function parameters
    return label_ds



if __name__ == '__main__':
    downloading_data(DATA_DIR,DATA_FNAME,DATA_URL)
