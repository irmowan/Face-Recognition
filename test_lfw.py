#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/5/16 上午10:27
# @Author  : irmo

import tensorflow as tf
import numpy as np
import cv2
import os.path
import transform
from tensorflow.contrib.slim.nets import vgg

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('model_name', '', """""")
tf.app.flags.DEFINE_string('lfw_data_dir', 'lfw_funneled', """""")
tf.app.flags.DEFINE_string('model_dir', 'train_data/casia_train/', """""")

FLAGS = tf.app.flags.FLAGS

threshold = 0.5
size = 6000
data_dir = 'dataset/lfw_funneled/'
image_output_dir = 'images/lfw_align/'
lfw_landmark_file = 'txt/lfw_landmark.txt'
pair_list_file = 'txt/pairs.txt'

restore_model = 'model.ckpt'
restore_step = 120000
restore_file = restore_model + '-' + str(restore_step)

no_landmark = 0


def load_lfw_landmark():
    dic = {}
    with open(lfw_landmark_file, 'r') as f:
        for line in f:
            info = line.split()
            filename = info[0]
            dic[filename] = [int(x) for x in info[-10:]]
    return dic


def test_forward(image):
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, end_points = vgg.vgg_16(image, num_classes=FLAGS.num_classes)
    extract_feature = 'vgg16/fc7'
    with tf.Session() as sess:
        saver = tf.train.saver(tf.global_variables())
        saver.restore(sess, restore_file)
        sess.run(tf.global_variables_initializer())
        end_point = sess.run([end_points])[0]
        print(end_point.shape)
        print(end_point)


def test_one_pair(image_file_pair, dic):
    image_file_0, image_file_1 = image_file_pair
    image_0 = cv2.imread(data_dir + image_file_0)
    image_1 = cv2.imread(data_dir + image_file_1)

    if image_file_0 in dic.keys():
        landmark_0 = dic[image_file_0]
    else:
        landmark_0 = None
    crop_image_0 = transform.img_process(image_0, landmark_0)
    assert crop_image_0.shape == (224, 224, 3)

    if image_file_1 in dic.keys():
        landmark_1 = dic[image_file_1]
    else:
        landmark_1 = None
    crop_image_1 = transform.img_process(image_1, landmark_1)
    assert crop_image_1.shape == (224, 224, 3)

    test_forward(crop_image_0)

    test_forward(crop_image_1)

    # cv2.imwrite(image_output_dir + image_file_0.split('/')[1], image_0)
    # cv2.imwrite(image_output_dir + image_file_1.split('/')[1], image_1)
    # cv2.imwrite(image_output_dir + image_file_0.split('/')[1][:-4] + '_crop.jpg', crop_image_0)
    # cv2.imwrite(image_output_dir + image_file_1.split('/')[1][:-4] + '_crop.jpg', crop_image_1)
    # feature_pair = []
    # for image in image_pair:
    #     landmark = get_landmark(image)
    #     image_align = align(image)
    #     feature_pair.append(get_feature(image_align))
    # d = cal_distance(feature_pair)
    # if d > threshold:
    #     return True
    # else:
    #     return False
    return True


def test():
    dic = load_lfw_landmark()

    with open(pair_list_file) as f:
        cnt = 0
        cnt_correct = 0
        cnt_false_positive = 0
        cnt_false_negative = 0
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
            answer = test_one_pair(image_pair, dic)
            cnt += 1
            # if (cnt == 10):
            #    break
            if answer == same:
                cnt_correct += 1
            elif answer and not same:
                cnt_false_positive += 1
            elif not answer and same:
                cnt_false_negative += 1
            # TODO: test only one case first
            break
        assert cnt == size
        print('Test completed.')
        print('Count = %d' % cnt)
        print('Correct   = %d, rate = %s' % (cnt_correct, format(cnt_correct / float(cnt), '6.2%')))
        print('F-p count = %d, rate = %s' % (cnt_false_positive, format(cnt_false_positive / float(cnt), '6.2%')))
        print('F-n count = %d, rate = %s' % (cnt_false_negative, format(cnt_false_negative / float(cnt), '6.2%')))


if __name__ == "__main__":
    test()
