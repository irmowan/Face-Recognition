#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/5/13 下午9:33
# @Author  : irmo

import tensorflow as tf
import multi_gpu_train
import cv2
import os.path

filename = 'casia_2.tfrecord'

file_queue = tf.train.string_input_producer([filename])

sess = tf.Session()
init = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())
images, labels = multi_gpu_train.read_and_decode(file_queue)
print('Run init')
sess.run(init)
tf.train.start_queue_runners(sess=sess)
print('Init complete')
path = 'test_tfrecord_2/'
for k in xrange(2):
    print('Run image')
    image, label = sess.run([images, labels])

    print('Image batch shape: ' + str(image.shape))
    print('Label batch shape: ' + str(label.shape))
    for i in xrange(image.shape[0]):
        cv2.imwrite(path + 'batch%02d_' % k + '%03d' % i + '_label_' + str(label[i]) + '.jpg', image[i])
