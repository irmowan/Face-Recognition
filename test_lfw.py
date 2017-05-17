#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/5/16 上午10:27
# @Author  : irmo

import tensorflow as tf
import numpy as np
import cv2
import os.path
import time
import transform
from tensorflow.contrib.slim.nets import vgg
from numpy import dot
from numpy.linalg import norm

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('model_name', '', """""")
tf.app.flags.DEFINE_string('lfw_data_dir', 'lfw_funneled', """""")
tf.app.flags.DEFINE_string('model_dir', 'train_data/casia_train/', """""")
tf.app.flags.DEFINE_integer('num_classes', 10575, """""")

FLAGS = tf.app.flags.FLAGS

threshold = 0.92
size = 6000
data_dir = 'dataset/lfw_funneled/'
image_output_dir = 'images/lfw_align/'
lfw_landmark_file = 'txt/lfw_landmark.txt'
pair_list_file = 'txt/pairs.txt'

restore_model = 'model.ckpt'
restore_step = 131000
restore_file = restore_model + '-' + str(restore_step)

extract_feature = 'vgg_16/fc6'

class TestLFW():
    def __init__(self):
        self.dic = {}
        self.sess = tf.Session()
        self.end_points = None
        self.images = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        self.dis_list = [] 
        
    def load_lfw_landmark(self):
        with open(lfw_landmark_file, 'r') as f:
            for line in f:
                info = line.split()
                filename = info[0]
                self.dic[filename] = [int(x) for x in info[-10:]]

    def def_net(self):
        with slim.arg_scope(vgg.vgg_arg_scope()):
            logits, end_points = vgg.vgg_16(self.images, num_classes=FLAGS.num_classes, 
                                            dropout_keep_prob=1.0, is_training=False)
        self.end_points = end_points

    def restore_model(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, FLAGS.model_dir + restore_file)
        self.sess.run(tf.global_variables_initializer())

    def get_features(self, image_0, image_1):
        images_0 = np.expand_dims(image_0, axis=0)
        images_1 = np.expand_dims(image_1, axis=0)
        images = np.concatenate((images_0, images_1), axis=0)
        end_point = self.sess.run([self.end_points], feed_dict={self.images:images})[0]
        features = end_point[extract_feature]
        feature_0, feature_1 = features[0][0][0], features[1][0][0]
        return feature_0, feature_1

    def test_one_pair(self, image_file_pair):
        image_file_0, image_file_1 = image_file_pair
        image_0 = cv2.imread(data_dir + image_file_0)
        image_1 = cv2.imread(data_dir + image_file_1)

        if image_file_0 in self.dic.keys():
            landmark_0 = self.dic[image_file_0]
        else:
            landmark_0 = None
        crop_image_0 = transform.img_process(image_0, landmark_0)
        assert crop_image_0.shape == (224, 224, 3)

        if image_file_1 in self.dic.keys():
            landmark_1 = self.dic[image_file_1]
        else:
            landmark_1 = None
        crop_image_1 = transform.img_process(image_1, landmark_1)
        assert crop_image_1.shape == (224, 224, 3)

        feature_0, feature_1 = self.get_features(crop_image_0, crop_image_1)
        distance = dot(feature_0, feature_1) / (norm(feature_0) * norm(feature_1))
        if distance > threshold:
            answer = True
        else:
            answer = False
        # cv2.imwrite(image_output_dir + image_file_0.split('/')[1], image_0)
        # cv2.imwrite(image_output_dir + image_file_1.split('/')[1], image_1)
        # cv2.imwrite(image_output_dir + image_file_0.split('/')[1][:-4] + '_crop.jpg', crop_image_0)
        # cv2.imwrite(image_output_dir + image_file_1.split('/')[1][:-4] + '_crop.jpg', crop_image_1)
        return answer, distance

    def test(self):
        self.load_lfw_landmark()
        self.def_net()
        self.restore_model()
        cnt = 0
        cnt_t_t = 0
        cnt_t_f = 0
        cnt_f_t = 0 
        cnt_f_f = 0
        start_time = time.time()
        with open(pair_list_file) as f:
            for line in f:
                info = line.split()
                image_pair = []
                same = None
                if len(info) == 3:
                    same = True
                    first = '/'.join([info[0], info[0] + '_' + '%04d' % int(info[1]) + '.jpg'])
                    second = '/'.join([info[0], info[0] + '_' + '%04d' % int(info[2]) + '.jpg'])
                    image_pair = [first, second]
                elif len(info) == 4:
                    same = False
                    first = '/'.join([info[0], info[0] + '_' + '%04d' % int(info[1]) + '.jpg'])
                    second = '/'.join([info[2], info[2] + '_' + '%04d' % int(info[3]) + '.jpg'])
                    image_pair = [first, second]
                else:
                    print('Line in the list error:')
                    print(line)
                    continue
                answer, distance = self.test_one_pair(image_pair)
                print(same, answer, distance)
                cnt += 1
                if answer and same:
                    cnt_t_t += 1
                elif not answer and not same:
                    cnt_f_f += 1
                elif answer and not same:
                    cnt_f_t += 1
                elif not answer and same:
                    cnt_t_f += 1
                self.dis_list.append(distance)
                if cnt % 500 == 0:
                    print('Test %4d/%4d pairs' % (cnt, size))
        duration = int(time.time() - start_time)
        assert cnt == size
        cnt_correct = cnt_t_t + cnt_f_f
        print('Test completed, use %d seconds.' % duration)
        print('Count = %d' % cnt)
        print('Correct = %d, rate = %s' % (cnt_correct, format(cnt_correct / float(cnt), '6.2%')))
        print('True,  guess True  = %d, rate = %s' % (cnt_t_t, format(cnt_t_t / float(cnt), '6.2%')))
        print('False, guess False = %d, rate = %s' % (cnt_f_f, format(cnt_f_f / float(cnt), '6.2%')))
        print('Flase, guess True  = %d, rate = %s' % (cnt_f_t, format(cnt_f_t / float(cnt), '6.2%')))
        print('True,  guess False = %d, rate = %s' % (cnt_t_f, format(cnt_t_f / float(cnt), '6.2%')))
        self.dis_list.sort()
        print('Medium distance: %.4f' % self.dis_list[cnt/2])

if __name__ == "__main__":
    t = TestLFW()
    t.test()
