import tensorflow as tf
import numpy as np


def load_casia_dataset(shape=(-1, 250, 250, 3)):
    """
    To see how to read dataset, refer to https://www.tensorflow.org/api_guides/python/io_ops#Readers
    :return:
    """

    file_names = ['./data/001.jpg', './data/002.jpg', './data/003.jpg']
    filename_queue = tf.train.string_input_producer(file_names)

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value, channels=3)
    image = np.array(image)
    image.reshape(shape)
    return image


if __name__ == "__main__":
    images = load_casia_dataset()
    print(images)
    images = load_casia_dataset()
    print(images)
    images = load_casia_dataset()
    print(images)

