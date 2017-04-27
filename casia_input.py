import tensorflow as tf
import numpy as np
from tensorflow.python.framework import dtypes

LIST_PATH = "list.txt"
NUM_EPOCHS = 100000
BATCH_SIZE = 50


def read_labeled_image_list(image_list_file):
    """
    Reading labeled images from a list
    Refer to http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels
    :param image_list_file: the path of the file
    :return: filenames and labels of the dataset
    """
    with open(image_list_file, 'r') as f:
        filenames = []
        labels = []
        for line in f:
            filename, label = line[:-1].split(' ')
            filenames.append(filename)
            labels.append(int(label))
        return filenames, labels


def read_images_from_disk(input_queue):
    """
    Read images from the disk
    :param input_queue: the input queue
    :return:
    """
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=3)
    image = tf.image.resize_images(example, [224, 224])
    image.set_shape((224, 224, 3))
    return image, label


if __name__ == "__main__":
    image_list, label_list = read_labeled_image_list(LIST_PATH)

    images = tf.convert_to_tensor(image_list, dtype=dtypes.string)
    labels = tf.convert_to_tensor(label_list, dtype=dtypes.int32)

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels], num_epochs=NUM_EPOCHS, shuffle=True)
    image, label = read_images_from_disk(input_queue)

    # Optional Image and Label Batching
    image_batch, label_batch = tf.train.batch([image, label], batch_size=BATCH_SIZE)
