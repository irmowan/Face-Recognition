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
# import vgg16_multi as vgg16

from tensorflow.contrib.slim.python.slim.nets import vgg

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'train_data/casia_train_multi',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 3,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('tfrecord_filename', 'casia_100.tfrecord',
                           """the name of the tfrecord""")
tf.app.flags.DEFINE_integer('batch_size', 32, """Batch size""")
tf.app.flags.DEFINE_integer('num_classes', 100, """Classes""")

TOWER_NAME = 'tower'
MOVING_AVERAGE_DECAY = 0.9999
NUM_IMAGES_PER_EPOCH = 4029
NUM_EPOCHS_PER_DECAY = 50

LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.01


def read_and_decode():
    """
    http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
    """
    filename = [FLAGS.tfrecord_filename]
    filename_queue = tf.train.string_input_producer(filename)

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
    image = tf.reshape(image, [224, 224, 3])
    image = tf.cast(image, tf.float32)
    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=FLAGS.batch_size,
                                            capacity=5000,
                                            num_threads=4,
                                            min_after_dequeue=1000)
    return images, labels


def tower_loss(scope):
    images, labels = read_and_decode()
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, end_points = vgg.vgg_16(images, num_classes=FLAGS.num_classes)
    _ = cal_loss(logits, labels)
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')
    for l in losses + [total_loss]:
        loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)
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
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_vars = (grad, v)
        average_grads.append(grad_and_vars)
    return average_grads


def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        num_batches_per_epoch = NUM_IMAGES_PER_EPOCH / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        print('Num batches per epoch = %d' % num_batches_per_epoch)
        print('Decay steps = %d' % decay_steps)
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
                    with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                        loss = tower_loss(scope)
                        tf.get_variable_scope().reuse_variables()
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        grads = optimizer.compute_gradients(loss)
                        tower_grads.append(grads)
        grads = average_gradients(tower_grads)
        summaries.append(tf.summary.scalar('learning_rate'), lr)
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name) + '/gradients', grad)

        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        variables_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variables_averages.apply(tf.trainable_variables())

        train_op = tf.group(apply_gradient_op, variables_averages_op)

        saver = tf.train.Saver(tf.global_variables())

        summary_op = tf.summary.merge(summaries)

        init = tf.global_variables_initializer()

        print('Create session...')
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))

        print('Init session...')
        sess.run(init)

        print('Start queue runners...')
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        print('Start training...')
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _ = sess.run([train_op])
            duration = time.time() - start_time

            if step % 10 == 0:
                loss_value, learning_rate = sess.run([loss, optimizer._learning_rate])
                num_images_per_step = FLAGS.batch_size * FLAGS.num_gpus
                images_per_sec = num_images_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus
                format_str = '%s: step %d, loss = %.4f, learning rate = %.4f (%.1f images/sec; %.3f sec/batch)'
                print(format_str % (datetime.now(), step, loss_value, learning_rate, images_per_sec, sec_per_batch))
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                
            if step % 500 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
