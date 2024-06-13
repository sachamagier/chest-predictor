import tensorflow as tf


def preprocess_images(path, resize_param=(224,224)):
        """
        Preprocesses a single image
        """
        #load the image from path
        image = tf.io.read_file(path)

        #decode
        img_tensor = tf.image.decode_png(image)[:, :, 0:1]

        # Convert grayscale image to RGB if necessary
        image_tensor = tf.image.grayscale_to_rgb(img_tensor)

        #resize the image based on the resize_param
        img_tensor = tf.image.resize(img_tensor, resize_param, method='nearest')

        #normalize the pixel values to [0,1]
        img_tensor /= 255.0

        return img_tensor



def load_and_preprocess_from_path_label(path, label):
    """
    Loads and preprocesses an image from a path.
    """
    return preprocess_images(path), label
