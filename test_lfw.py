#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/5/16 上午10:27
# @Author  : irmo

import tensorflow as tf
import numpy as np
import cv2
import os.path
import time
import pickle
import transform
from tensorflow.contrib.slim.nets import vgg, resnet_v1
from numpy import dot
from numpy.linalg import norm

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('lfw_data_dir', 'dataset/lfw_funneled/', """""")
tf.app.flags.DEFINE_integer('num_classes', 10575, """""")

FLAGS = tf.app.flags.FLAGS

model_name = 'vgg_16'
restore_model = 'model.ckpt'
restore_step = 135000
choose_feature = 'fc7'

size = 6000
image_output_dir = 'images/lfw_align/'
lfw_landmark_file = 'txt/lfw_landmark.txt'
pair_list_file = 'txt/pairs.txt'
model_dir = 'train_data/casia_' + model_name + '/'
restore_file = restore_model + '-' + str(restore_step)
features_dict = {
    'fc6': 'vgg_16/fc6',
    'fc7': 'vgg_16/fc7',
    'fc8': 'vgg_16/fc8',
    'res': 'resnet_v1_101/block2/unit_2/bottleneck_v1/conv3'
}
extract_feature = features_dict[choose_feature]
features_dir = 'features'
output_feature_path = os.path.join(features_dir,'_'.join(('lfw', model_name, 'step' + str(restore_step), choose_feature)) + '.txt')

class TestLFW():
    def __init__(self):
        self.dic = {}
        self.sess = tf.Session()
        self.end_points = None
        self.images = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))

    def load_lfw_landmark(self):
        with open(lfw_landmark_file, 'r') as f:
            for line in f:
                info = line.split()
                filename = info[0]
                self.dic[filename] = [int(x) for x in info[-10:]]

    def def_net(self):
        if model_name == 'vgg_16':
            with slim.arg_scope(vgg.vgg_arg_scope()):
                _, end_points = vgg.vgg_16(self.images, num_classes=FLAGS.num_classes,
                                           dropout_keep_prob=1.0, is_training=False)
        elif model_name == 'resnet_v1':
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                _, end_points = resnet_v1.resnet_v1_101(self.images, num_classes=FLAGS.num_classes)
        self.end_points = end_points

    def restore_model(self):
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess, model_dir + restore_file)

    def get_features(self, image_0, image_1):
        images_0 = np.expand_dims(image_0, axis=0)
        images_1 = np.expand_dims(image_1, axis=0)
        images = np.concatenate((images_0, images_1), axis=0)
        end_point = self.sess.run([self.end_points], feed_dict={self.images: images})[0]
        features = end_point[extract_feature]
        feature_0, feature_1 = features[0][0][0], features[1][0][0]
        return feature_0, feature_1

    def test_one_pair(self, image_file_pair):
        image_file_0, image_file_1 = image_file_pair
        image_0 = cv2.imread(FLAGS.lfw_data_dir + image_file_0)
        image_1 = cv2.imread(FLAGS.lfw_data_dir + image_file_1)
        assert image_0 is not None
        assert image_1 is not None
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
        assert feature_0.ndim == 1
        assert feature_1.ndim == 1
        # feature_0 = feature_0 / norm(feature_0)
        # feature_1 = feature_1 / norm(feature_1)
        # similarity = norm(feature_0-feature_1)
        similarity = dot(feature_0, feature_1) / (norm(feature_0) * norm(feature_1))
        # cv2.imwrite(image_output_dir + image_file_0.split('/')[1], image_0)
        # cv2.imwrite(image_output_dir + image_file_1.split('/')[1], image_1)
        # cv2.imwrite(image_output_dir + image_file_0.split('/')[1][:-4] + '_crop.jpg', crop_image_0)
        # cv2.imwrite(image_output_dir + image_file_1.split('/')[1][:-4] + '_crop.jpg', crop_image_1)
        return feature_0, feature_1, similarity

    def generate_image_pairs(self):
        image_pairs = []
        with open(pair_list_file) as f:
            for line in f:
                info = line.split()
                same = None
                image_pair = {}
                if len(info) == 3:
                    same = True
                    first = '/'.join([info[0], info[0] + '_' + '%04d' % int(info[1]) + '.jpg'])
                    second = '/'.join([info[0], info[0] + '_' + '%04d' % int(info[2]) + '.jpg'])
                elif len(info) == 4:
                    same = False
                    first = '/'.join([info[0], info[0] + '_' + '%04d' % int(info[1]) + '.jpg'])
                    second = '/'.join([info[2], info[2] + '_' + '%04d' % int(info[3]) + '.jpg'])
                else:
                    print('Line in the list error:')
                    print(line)
                    continue
                image_pair['files'] = [first, second]
                image_pair['ground_truth'] = same
                image_pairs.append(image_pair)
        return image_pairs

    def search_threshold(self, sorted_pairs):
        correct = size / 2
        t_t = size / 2
        f_f = 0
        best_correct = correct
        best_threshold = 0.0
        best_t_t = t_t
        best_f_f = f_f
        for image_pair in sorted_pairs:
            if image_pair['ground_truth'] is True:
                correct -= 1
                t_t -= 1
            else:
                correct += 1
                f_f += 1
            if correct > best_correct:
                best_correct = correct
                best_threshold = image_pair['similarity']
                best_t_t, best_f_f = t_t, f_f
        return best_correct, best_threshold, best_t_t, best_f_f

    def test(self):
        self.load_lfw_landmark()
        self.def_net()
        self.restore_model()
        image_pairs = self.generate_image_pairs()
        assert len(image_pairs) == size

        start_time = time.time()
        print('Begin test...')
        for idx, image_pair in enumerate(image_pairs):
            feature_0, feature_1, similarity = self.test_one_pair(image_pair['files'])
            image_pair['features'] = [feature_0, feature_1]
            image_pair['similarity'] = similarity
            if (idx + 1) % 500 == 0:
                print('Tested %4d/%4d pairs' % (idx + 1, size))
            
        duration = int(time.time() - start_time)
        print('Test completed, use %d seconds.' % duration)

        sorted_pairs = sorted(image_pairs, key=lambda x: x['similarity'])

        print('Searching for best threshold...')
        best_correct, best_threshold, best_t_t, best_f_f = self.search_threshold(sorted_pairs)
        print('Choose threshold: %.4f' % best_threshold)

        print('Size = %d, Correct = %4d, rate = %s' % (size, best_correct, format(best_correct / float(size), '6.2%')))
        print('True,  guess True  = %4d, rate = %s' % (best_t_t, format(best_t_t / float(size), '6.2%')))
        print('False, guess False = %4d, rate = %s' % (best_f_f, format(best_f_f / float(size), '6.2%')))
        print('Storing features...')
        pickle.dump(image_pairs, open(output_feature_path, 'w'))
        print('Features stored at %s' % output_feature_path)

if __name__ == "__main__":
    if not tf.gfile.Exists(features_dir):
        tf.gfile.MakeDirs(features_dir)
    t = TestLFW()
    t.test()
