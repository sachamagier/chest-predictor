import pandas as pd
import tensorflow as tf
import os
from pathlib import Path
from chest_predictor.params import *


def load_labels(DATA_DIR, all_image_paths):
    data_root = Path(os.path.join(DATA_DIR, 'resized_dataset'))
    labels_df = pd.read_csv(data_root /"Data_Entry_2017.csv")
    labels_df.set_index('Image Index', inplace=True)

    # Create list of all image labels
    all_image_labels = labels_df.loc[[os.path.basename(path) for path in all_image_paths], 'Finding Labels'].values

    return labels_df, all_image_labels



def encoding_labels(label_names, all_image_labels):
    # Create a StringLookup layer to map labels to indices
    string_lookup = tf.keras.layers.StringLookup(vocabulary=label_names, num_oov_indices=0)
    # Vectorized encoding of labels
    def encoded_labels(labels):
        # Split the string of labels into a list of individual labels
        split_labels = tf.strings.split(labels, '|')
         # Map each label to an index using the StringLookup layer
        encoded_labels = string_lookup(split_labels)
       # One-hot encode the indices using the tf.one_hot method
        # and sum the one-hot vectors using the tf.reduce_sum method
        # to create a single one-hot vector that represents the presence
        # or absence of each finding in the chest X-ray image
        one_hot_encoded = tf.reduce_sum(tf.one_hot(encoded_labels, depth=len(label_names)), axis=0)
        return one_hot_encoded
    ## Apply encoding to all labels in a vectorized manner
    # using the tf.stack method to create a tensor of one-hot vectors
    encoded_values = tf.stack([encoded_labels(labels) for labels in all_image_labels])
    # Convert encoded values to tensor slices for the dataset
    label_ds = tf.data.Dataset.from_tensor_slices(encoded_values)

    return label_ds



### the next code could be used if necessary in the future

# #to have a vocabulary
# label_names = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
#                'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
#                'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'No Finding']
# string_lookup = tf.keras.layers.StringLookup(vocabulary=label_names, num_oov_indices=0)
# vocab = string_lookup.get_vocabulary()

# Display the vocabulary and their corresponding indices
# for i, label in enumerate(vocab):
#     print(f"Index {i}: {label}")

if __name__ == '__main__':

    load_labels(DATA_DIR)
    print('labels loaded 💪')
