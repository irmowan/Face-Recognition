#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/5/12 下午4:10
# @Author  : irmo

import input
import numpy as np
import tensorflow as tf
import transform

INPUT_LIST = 'casia.txt'
OUTPUT_FILE = 'casia.tfrecord'
NUM_CLASSES = 10575

if __name__ == "__main__":
    input_queue = input.generate_input_queue(INPUT_LIST)
    print('Size of the queue is ' + str(len(input_queue)))
    images = []
    labels = []
    label_map = {}
    writer = tf.python_io.TFRecordWriter(OUTPUT_FILE)
    while len(input_queue) > 0:
        image, label_idx, landmark = input.read_image_from_disk(input_queue)
        if label_idx in label_map.keys():
            new_label = label_map[label_idx]
        else:
            new_label = len(label_map)
            label_map[label_idx] = new_label
        if image.ndim == 2:
            image = np.dstack([image] * 3)
        cropped_image = transform.img_process(image, landmark)
        image_raw = cropped_image.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[new_label])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
        }))
        writer.write(example.SerializeToString())
        if len(input_queue) % 1000 == 0:
            print('Left %d images to pack' % len(input_queue))
    print(label_map)
    writer.close()
