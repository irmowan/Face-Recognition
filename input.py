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

import re
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from scipy import misc
import os
import random
import transform
import cv2

tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'data/',
                           """Path to the CASIA data directory.""")
tf.app.flags.DEFINE_integer('num_classes', 100,
                            """Number of classes""")
FLAGS = tf.app.flags.FLAGS

LIST_FILE = "casia_100.txt"
IMAGE_SIZE = 224
TOWER_NAME = 'tower'
input_queue = None

NUM_IMAGES_PER_EPOCH_FOR_TRAIN = 4029  # For casia_100


# NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2

# Constants
# MOVING_AVERAGE_DECAY = 0.9999
# NUM_EPOCHS_PER_DECAY = 350.0
# LEARNING_RATE_DECAY_FACTOR = 0.1
# INITIAL_LEARNING_RATE = 0.1


def read_image_from_disk(input_queue):
    """
    :param input_queue:
    :return:
    """
    element = input_queue.pop(0)
    file_path = element[0]
    label = element[1]
    landmark = element[2]
    # image = misc.imread(FLAGS.data_dir + file_path, mode='RGB')
    image = cv2.imread(FLAGS.data_dir + file_path)
    return image, label, landmark


def generate_input_queue(image_list_file=LIST_FILE, shuffle=True):
    """
    :param batch_size:
    :param max_num_epochs:
    :param num_preprocess_threads:
    :param shuffle:
    :return:
    """

    def read_labeled_image_list(image_list_file):
        """
        Reading labeled images from a list
        :param image_list_file: the path of the file
        :return: filenames and labels of the dataset
        """
        with open(image_list_file, 'r') as f:
            image_list = []
            label_list = []
            landmark_list = []
            for idx, line in enumerate(f):
                filename, label = line[:-1].split(' ')[:2]
                l = line[:-1].split(' ')[6:]
                landmark = [int(x) for x in l]
                if os.path.exists(FLAGS.data_dir + filename):
                    image_list.append(filename)
                    label_list.append(int(label))
                    landmark_list.append(landmark)
                else:
                    print('File not found: ' + filename)
        return image_list, label_list, landmark_list

    print('Generate input queue...')
    image_list, label_list, landmark_list = read_labeled_image_list(image_list_file)
    input_queue = zip(image_list, label_list, landmark_list)
    if shuffle:
        random.shuffle(input_queue)
    # images = ops.convert_to_tensor(image_list, dtype=tf.string)
    # labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
    # labels = tf.one_hot(label_list, depth=NUM_CLASSE, on_value=1.0, off_value=0.0, axis=-1)
    # images = image_list
    # labels = label_list
    # print(type(images))
    # print(type(labels))
    # labels = np.zeros((len(images), FLAGS.num_classes))
    # labels[np.arange(len(images)), label_list] = 1
    # input_queue = tf.train.slice_input_producer(
    #    [images, labels], num_epochs=max_num_epochs, shuffle=shuffle)
    return input_queue


def center_crop(image):
    crop = 224
    y, x, _ = image.shape
    startx = x // 2 - (crop // 2)
    starty = y // 2 - (crop // 2)
    return image[starty:starty + crop, startx:startx + crop, :]


def read_casia():
    global input_queue

    images = []
    labels = []
    for i in range(FLAGS.batch_size):
        if input_queue is None or len(input_queue) == 0:
            input_queue = generate_input_queue()
        image, label_idx, landmark = read_image_from_disk(input_queue)
        if image.ndim == 2:
            image = np.dstack([image] * 3)
        croped_image = transform.img_process(image, landmark)
        images.append(croped_image)
        label = np.zeros(FLAGS.num_classes)
        label[label_idx] = 1
        labels.append(label)
        # except Exception e:
        #     print('Error: ' + str(e))
        #     i = i - 1
    image_batch = np.array([x for x in images])
    label_batch = np.array([x for x in labels])
    # print(len(images_and_labels))
    # image_batch, label_batch = tf.train.batch_join(
    #     images_and_labels,
    #     batch_size=FLAGS.batch_size,
    #     capacity=4 * num_preprocess_threads * FLAGS.batch_size,
    #     allow_smaller_final_batch=False
    # )

    # print('Return batch size: ' + str(image_batch.shape) + ' ' + str(label_batch.shape))
    # sess = tf.Session()
    # with sess.as_default():
    #     images = tf.cast(image_batch, tf.float32).eval()
    #     labels = tf.cast(label_batch, tf.float32).eval()
    return image_batch, label_batch


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def _activation_summary(x):
    """
    Helper to create summaries for activations.
    Create a summary that provides a histogram of activations
    Creaete a summary that measures the sparsity of activations

    :param x:
    :return:
    """
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

#
# def _variable_on_cpu(name, shape, initializer):
#     with tf.device('/cpu:0'):
#         var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
#     return var
#
#
# def _variable_with_weight_decay(name, shape, stddev, wd):
#     var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
#     if wd is not None:
#         weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
#         tf.add_to_collection('losses', weight_decay)
#     return var
#
#
# def loss(logits, labels):
#     labels = tf.cast(labels, tf.int32)
#     cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
#         labels=labels, logits=logits, name='cross_entropy_per_example'
#     )
#     cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
#     tf.add_to_collection('losses', cross_entropy_mean)
#     return tf.add_n(tf.get_collection('losses'), name='total_loss')


# def inference(images):
#     # conv1
#     with tf.variable_scope('conv1') as scope:
#         kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64], stddev=5e-2, wd=0.0)
#         conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
#         print('conv 1 shape:' + str(conv.get_shape()))
#         biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
#
#         pre_activation = tf.nn.bias_add(conv, biases)
#         print(pre_activation.get_shape())
#         conv1 = tf.nn.relu(pre_activation, name=scope.name)
#         _activation_summary(conv1)
#
#     # pool1
#     pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
#                            padding='SAME', name='pool1')
#     norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
#     # conv2
#     with tf.variable_scope('conv2') as scope:
#         kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0)
#         conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
#         biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv2 = tf.nn.relu(pre_activation, name=scope.name)
#         _activation_summary(conv2)
#
#     norm2 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
#     pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
#                            padding='SAME', name='pool2')
#     with tf.variable_scope('local3') as scope:
#         reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
#         dim = reshape.get_shape()[1].value
#         print('local3 reshape shape: ' + str(reshape.get_shape()))
#         weights = _variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
#         biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
#         local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
#         _activation_summary(local3)
#     with tf.variable_scope('local4') as scope:
#         weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
#         biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
#         local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
#         _activation_summary(local4)
#
#     with tf.variable_scope('softmax_linear') as scope:
#         weights = _variable_with_weight_decay('weights', [192, FLAGS.num_classes],
#                                               stddev=1 / 192.0, wd=0.0)
#         biases = _variable_on_cpu('biases', [FLAGS.num_classes], tf.constant_initializer(0.0))
#         softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
#         _activation_summary(softmax_linear)
#     return softmax_linear

# def train(total_loss, global_step):
#     num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
#
#     decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
#
#     # Delay the learning rate exponentially based on the number of steps
#     lr = tf.train.exponential_decay(
#         INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=False)
#     tf.summary.scalar('learning_rate', lr)
#
#     # Generate moving averages of all losses and associated summaries
#     loss_averages_op = _add_loss_summaries(total_loss)
#
#     # Compute gradients
#     with tf.control_dependencies([loss_averages_op]):
#         opt = tf.train.GradientDescentOptimizer(lr)
#         grads = opt.compute_gradients(total_loss)
#
#     # Apply gradients
#     apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
#
#     # Add histogram
#     for grad, var in grads:
#         if grad is not None:
#             tf.summary.histogram(var.op.name + '/gradients', grad)
#
#     # Track the moving averages of all trainable variables
#     variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
#     variable_averages_op = variable_averages.apply(tf.trainable_variables())
#
#     with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
#         train_op = tf.no_op(name='train')
#
#     return train_op
#
#
# if __name__ == "__main__":
#     pass
