#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/5/13 下午9:33
# @Author  : irmo

import tensorflow as tf
import multi_gpu_train
import cv2
import os.path

FLAGS = tf.app.flags.FLAGS

file_queue = tf.train.string_input_producer([FLAGS.tfrecord_filename])

sess = tf.Session()
init = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())
sess.run(init)

for k in range(2):
    images, labels = multi_gpu_train.read_and_decode(file_queue)
    image, label = sess.run([images, labels])

    print('Image batch shape: ' + str(image.shape))
    print('Label batch shape: ' + str(label.shape))
    path = 'test_tfrecord'
    for i in xrange(image.shape[0]):
        cv2.imwrite(os.path.join(path, 'batch%d' % k + '%02d' % i + '.jpg'), image[i])
