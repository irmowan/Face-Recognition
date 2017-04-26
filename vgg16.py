#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
import os
import sys
import numpy as np
import time
from scipy.misc import imread, imresize


def conv_layers(net_in):
    """
    Define the convolutional network
    :param net_in:
    :return:
    """
    network = tl.layers.Conv2dLayer(net_in,
                                    act=tf.nn.relu,
                                    # 64 features for each 3x3 patch
                                    shape=[3, 3, 3, 64],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv1_1')
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    # 64 features for each 3x3 patch
                                    shape=[3, 3, 64, 64],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv1_2')
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool1')
    """ conv2 """
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    # 128 features for each 3x3 patch
                                    shape=[3, 3, 64, 128],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv2_1')
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    # 128 features for each 3x3 patch
                                    shape=[3, 3, 128, 128],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv2_2')
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool2')
    """ conv3 """
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    # 256 features for each 3x3 patch
                                    shape=[3, 3, 128, 256],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv3_1')
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    # 256 features for each 3x3 patch
                                    shape=[3, 3, 256, 256],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv3_2')
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    # 256 features for each 3x3 patch
                                    shape=[3, 3, 256, 256],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv3_3')
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool3')
    """ conv4 """
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    # 512 features for each 3x3 patch
                                    shape=[3, 3, 256, 512],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv4_1')
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    # 512 features for each 3x3 patch
                                    shape=[3, 3, 512, 512],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv4_2')
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    # 512 features for each 3x3 patch
                                    shape=[3, 3, 512, 512],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv4_3')
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool4')
    """ conv5 """
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    # 512 features for each 3x3 patch
                                    shape=[3, 3, 512, 512],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv5_1')
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    # 512 features for each 3x3 patch
                                    shape=[3, 3, 512, 512],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv5_2')
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    # 512 features for each 3x3 patch
                                    shape=[3, 3, 512, 512],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv5_3')
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool5')
    return network


def fc_layers(net, nums_identity):
    """
    Define Full-connect layers
    :param net: network
    :param nums_identity: the nums of the last layer
    :return:
    """
    network = tl.layers.FlattenLayer(net, name='flatten')
    network = tl.layers.DenseLayer(network,
                                   n_units=4096,
                                   act=tf.nn.relu,
                                   name='fc1_relu')
    network = tl.layers.DenseLayer(network,
                                   n_units=4096,
                                   act=tf.nn.relu,
                                   name='fc2_relu')
    network = tl.layers.DenseLayer(network,
                                   n_units=nums_identity,
                                   act=tf.identity,
                                   name='fc3_relu')
    return network


if __name__ == "__main__":
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

    net_in = tl.layers.InputLayer(x, name='input_layer')
    net_cnn = conv_layers(net_in)
    network = fc_layers(net_cnn, 1000)

    y = network.outputs
    probs = tf.nn.softmax()
    y_op = tf.argmax(tf.nn.softmax(y), 1)
    cost = tl.cost.cross_entropy(y, y_, name='cost')

    correct_prediction = tf.equal(
        tf.cast(tf.argmax(y, 1), tf.float32), tf.cast(y_, tf.float32))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # sess.run()
    tl.layers.initialize_global_variables(sess)
    network.print_params()
    network.print_layers()

    if not os.path.isfile("vgg16_weights.npz"):
        print("Please download vgg16_weights.npz from : http://www.cs.toronto.edu/~frossard/post/vgg16/")
        exit()
    npz = np.load('vgg16_weights.npz')

    params = []
    for val in sorted(npz.items()):
        print(" Loading %s" % str(val[1].shape))
        params.append(val[1])

    tl.files.assign_params(sess, params, network)
