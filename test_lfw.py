#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/5/16 上午10:27
# @Author  : irmo

import tensorflow as tf
import numpy as np
import cv2
import os.path

import transform

tf.app.flags.DEFINE_string('model_name', '', """""")
tf.app.flags.DEFINE_string('lfw_data_dir', 'lfw_funneled', """""")

FLAGS = tf.app.flags.FLAGS

threshold = 0.5
size = 6000
image_output_dir = 'images/lfw_align/'
lfw_landmark_file = 'lfw_landmark.txt'
pair_list_file = 'pairs.txt'


def load_lfw_landmark():
    dic = {}
    with open(lfw_landmark_file, 'r') as f:
        for line in f:
            info = line.split()
            filename = info[0]
            dic[filename] = [int(x) for x in info[-10:]]
    return dic


def test_one_pair(image_file_pair, dic):
    image0 = cv2.imread(image_file_pair[0])
    image1 = cv2.imread(image_file_pair[1])
    landmark0 = dic[image0]
    landmark1 = dic[image1]
    crop_image0 = transform.img_process(image0, landmark0)
    crop_image1 = transform.img_process(image1, landmark1)
    cv2.imwrite(image_output_dir + image_file_pair[0], image0)
    cv2.imwrite(image_output_dir + image_file_pair[1], image1)
    cv2.imwrite(image_output_dir + image_file_pair[0][:-4] + '_crop.jpg', crop_image0)
    cv2.imwrite(image_output_dir + image_file_pair[1][:-4] + '_crop.jpg', crop_image1)
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
            if answer == same:
                cnt_correct += 1
            elif answer and not same:
                cnt_false_positive += 1
            elif not answer and same:
                cnt_false_negative += 1
        print('Test completed.')
        print('Correct count    = %d, rate = %.3f' % (cnt_correct, cnt_correct / size))
        print('F-positive count = %d, rate = %.3f' % (cnt_false_positive, cnt_false_positive / size))
        print('F-negative count = %d, rate = %.3f' % (cnt_false_negative, cnt_false_negative / size))


if __name__ == "__main__":
    test()
