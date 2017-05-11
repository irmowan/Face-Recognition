#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/4/28 下午5:33
# @Author  : irmo
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
import vgg16 as vgg16

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/casia_train_multi',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
TOWER_NAME = 'tower'


def tower_loss(scope, vgg):
    # images_batch, labels_batch = input.read_casia()
    # logits = vgg.prob(images_batch)

    # Build the portion of the Graph calculating the losses.
    # _ = input.loss(logits, labels_batch)
    losses = tf.get_collection('losses', scope)

    total_loss = tf.add_n(losses, name='total_loss')

    # for l in losses + [total_loss]:
    #     loss_name = re.sub('%s_[0-9]*/' % input.TOWER_NAME, '', l.op.name)
    #     tf.summary.scalar(loss_name, l)
    # print('Completed')
    return total_loss


def average_gradients(tower_grads):
    print('Calculating average gradients........')
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


# def build_op(self):
#     trainable_variables = tf.trainable_variables()
#     grads = tf.gradients(self.cost, trainable_variables)
#     # optimizer = tf.train.AdadeltaOptimizer(self.lrn_rate, 0.9)
#
#     apply_op = optimizer.apply_gradients(
#         zip(grads, trainable_variables), name='train_step')
#
#     self.train_op_list = [apply_op]
#     self.train_op = tf.group(*self.train_op_list)


def train():
    # with tf.Graph().as_default(), tf.device('/cpu:0'):
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=True)
    vgg = vgg16.VGG16(trainable=True)

    optimizer = tf.train.AdadeltaOptimizer(vgg.lrn_rate, 0.9)
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in xrange(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (input.TOWER_NAME, i)) as scope:
                    loss = tower_loss(scope, vgg)
                    tf.get_variable_scope().reuse_variables()
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)
    grads = average_gradients(tower_grads)
    apply_op = optimizer.apply_gradients(
        zip(grads, tf.trainable_variables()), global_step=global_step)
    train_op_list = [apply_op]
    train_op = tf.group(*train_op_list)
    # print('Calculate the mean of each gradient')
    # print('Tower grads type:' + str(type(tower_grads)))
    # grads = average_gradients(tower_grads)
    # summaries.append(tf.summary.scalar('learning_rate', lr))
    # print('Add a histograms to track the learning rate')
    # for grad, var in grads:
    #     if grad is not None:
    #         summaries.append(tf.summary.histogram(
    #             var.op.name + '/gradients', grad))

    # print('Apply the gradients to adjust the shared variables')
    # # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    # for var in tf.trainable_variables():
    #     summaries.append(tf.summary.histogram(var.op.name, var))

    # print('Define moving average')
    # variable_averages = tf.train.ExponentialMovingAverage(model.MOVING_AVERAGE_DECAY, global_step)
    # variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # train_op = tf.group(apply_gradient_op, variables_averages_op)
    saver = tf.train.Saver(tf.global_variables())
    # summary_op = tf.summary.merge(summaries)

    init = tf.global_variables_initializer()
    print('Create session')
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))

    print('Init session...')
    sess.run(init)

    print('Start training...')
    # tf.train.start_queue_runners(sess=sess)

    # summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    for step in xrange(FLAGS.max_steps):
        # print('Step %d' % step)
        start_time = time.time()
        images, labels = input.read_casia()
        _ = sess.run(train_op, feed_dict={vgg.imgs: images, vgg.labels: labels})
        duration = time.time() - start_time

        if step % 10 == 0:
            fc7, pred, loss_value = sess.run([vgg.fc7, vgg.prob, vgg.cost],
                                             feed_dict={vgg.imgs: images, vgg.labels: labels})
            num_images_per_step = FLAGS.batch_size * FLAGS.num_gpus
            images_per_sec = num_images_per_step / duration
            sec_per_batch = duration / FLAGS.num_gpus
            format_str = '%s: step %d, loss = %.4f (%.1f images/sec; %.3f sec/batch)'
            print(format_str % (datetime.now(), step, loss_value, images_per_sec, sec_per_batch))

        # if step % 100 == 0:
        #     summary_str = sess.run(summary_op)
        #     summary_writer.add_summary(summary_str, step)

        if step % 100 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
        if step == 10000:
            vgg.lrn_rate /= 10
        if step == 100000:
            vgg.lrn_rate /= 10
        if step == 1000000:
            vgg.lrn_rate /= 10


def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
