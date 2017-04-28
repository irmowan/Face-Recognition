#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/4/28 下午2:25
# @Author  : irmo

"""
Refer to
https://github.com/davidsandberg/facenet/blob/master/src/facenet.py
http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels
https://www.tensorflow.org/programmers_guide/reading_data
https://www.tensorflow.org/api_guides/python/io_ops#Input_pipeline
https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_input.py
https://gist.github.com/eerwitt/518b0c9564e500b4b50f
"""

import tensorflow as tf
from tensorflow.python.framework import ops

LIST_FILE = "casia.txt"


def read_labeled_image_list(image_list_file):
    """
    Reading labeled images from a list
    :param image_list_file: the path of the file
    :return: filenames and labels of the dataset
    """
    with open(image_list_file, 'r') as f:
        image_list = []
        label_list = []
        for line in f:
            filename, label = line[:-1].split(' ')[:2]
            image_list.append(filename)
            label_list.append(int(label))
        return image_list, label_list


def read_image_from_disk(input_queue):
    """
    :param input_queue:
    :return:
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    return example, label


def read_casia(batch_size=64, max_num_epochs=10000, num_preprocess_threads=16, shuffle=True):
    """
    :param batch_size:
    :param max_nrof_epochs:
    :param nrof_preprocess_threads:
    :param shuffle:
    :return:
    """
    image_list, label_list = read_labeled_image_list(image_list_file=LIST_FILE)
    images = ops.convert_to_tensor(image_list, dtype=tf.string)
    labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
    input_queue = tf.train.slice_input_producer(
        [images, labels], num_epochs=max_num_epochs, shuffle=shuffle)

    images_and_labels = []
    for _ in range(num_preprocess_threads):
        image, label = read_image_from_disk(input_queue)
        image = tf.random_crop(image, [224, 224, 3])
        images_and_labels.append([image, label])

    image_batch, label_batch = tf.train.batch_join(
        images_and_labels,
        batch_size=batch_size,
        capacity=4 * num_preprocess_threads * batch_size,
        allow_smaller_final_batch=True
    )
    return image_batch, label_batch


if __name__ == "__main__":
    pass
