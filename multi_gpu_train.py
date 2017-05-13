#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/5/12
# @Author  : irmo
# This version is imitating cifar10_multi_gpu_train.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

from six.moves import xrange
import numpy as np
import tensorflow as tf
import input
import vgg16_multi as vgg16

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'train_data/casia_train_multi',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 3,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('tfrecord_filename', 'casia100.tfrecord',
                           """the name of the tfrecord""")

TOWER_NAME = 'tower'
MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1


def read_and_decode(filename_queue):
    """
    http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)

    image.set_shape([224, 224, 3])
    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=FLAGS.batch_size,
                                            capacity=30,
                                            num_threads=2,
                                            min_after_dequeue=10)
    return images, labels


def tower_loss(scope, vgg):
    images, labels = read_and_decode(FLAGS.tfrecord_filename)
    vgg.imgs = tf.cast(images, tf.float32)
    vgg.labels = tf.cast(labels, tf.float32)
    logits = vgg.predictions
    _ = cal_loss(logits, vgg.labels)
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')
    return total_loss


def cal_loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            if not (g is None):
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
        if len(grads) > 0:
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_vars = (grad, v)
            average_grads.append(grad_and_vars)
    return average_grads


def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=True)
        num_batches_per_epoch = input.NUM_IMAGES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        vgg = vgg16.VGG16(trainable=True)
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(lr)
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (input.TOWER_NAME, i)) as scope:
                        loss = tower_loss(scope, vgg)
                        tf.get_variable_scope().reuse_variables()
                        grads = optimizer.compute_gradients(loss)
                        tower_grads.append(grads)
        grads = average_gradients(tower_grads)
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
        variables_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variables_averages.apply(tf.trainable_variables())
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # apply_op = optimizer.apply_gradients(
        #     zip(grads, tf.trainable_variables()), global_step=global_step)
        # train_op_list = [apply_op]
        # train_op = tf.group(*train_op_list)

        saver = tf.train.Saver(tf.global_variables())

        init = tf.global_variables_initializer()
        print('Create session...')
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))

        print('Init session...')
        sess.run(init)

        print('Start training...')
        tf.train.start_queue_runners(sess=sess)
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            images, labels = input.read_casia()
            _ = sess.run([train_op], feed_dict={vgg.imgs: images, vgg.labels: labels})
            duration = time.time() - start_time

            if step % 10 == 0:
                loss_value = sess.run(loss, feed_dict={vgg.imgs: images, vgg.labels: labels})
                num_images_per_step = FLAGS.batch_size * FLAGS.num_gpus
                images_per_sec = num_images_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus
                format_str = '%s: step %d, loss = %.4f (%.1f images/sec; %.3f sec/batch)'
                print(format_str % (datetime.now(), step, loss_value, images_per_sec, sec_per_batch))
            if step % 100 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
