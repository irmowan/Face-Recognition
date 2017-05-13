#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/5/12 下午4:10
# @Author  : irmo

import input
import numpy as np
import tensorflow as tf
import transform

INPUT_LIST = 'casia_100.txt'
OUTPUT_FILE = 'casia_100.tfrecord'
NUM_CLASSES = 100

if __name__ == "__main__":
    input_queue = input.generate_input_queue('casia_100.txt')
    print('Size of the queue is ' + str(len(input_queue)))
    images = []
    labels = []
    writer = tf.python_io.TFRecordWriter(OUTPUT_FILE)
    while len(input_queue) > 0:
        image, label_idx, landmark = input.read_image_from_disk(input_queue)
        if image.ndim == 2:
            image = np.dstack([image] * 3)
        cropped_image = transform.img_process(image, landmark)
        image_raw = image.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_idx])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
        }))
        writer.write(example.SerializeToString())
    writer.close()