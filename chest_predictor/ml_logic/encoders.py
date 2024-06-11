import pandas as pd
import tensorflow as tf

data_path = "../raw_data/encoded_labels.csv"

def load_and_encode_labels(data_path):
    """
    Load encoded labels from a CSV file. and then transform them into a tensor

    Args:
    - data_path (str): Path to the CSV file containing encoded labels
    """
    # Read the CSV file
    df_labels = pd.read_csv(data_path)

    # Convert the labels into a TensorFlow tensor
    encoded_values = tf.convert_to_tensor(df_labels.values, dtype=tf.float32)
    label_ds = tf.data.Dataset.from_tensor_slices(encoded_values)
    # Depending on if we are doing this in another .py or not we can add the nextg line. It creates a Dataset with image paths and encoded labels
    #path_label_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, encoded_values)) --> add all_image_paths to the function parameters
    return label_ds



### the next code could be used if necessary in the future

#to have a vocabulary
label_names = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
               'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
               'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'No Finding']
string_lookup = tf.keras.layers.StringLookup(vocabulary=label_names, num_oov_indices=0)
vocab = string_lookup.get_vocabulary()

# Display the vocabulary and their corresponding indices
for i, label in enumerate(vocab):
    print(f"Index {i}: {label}")
