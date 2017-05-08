#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/5/8 下午3:59
# @Author  : irmo
import os

INPUT = 'casia.txt'
OUTPUT = 'casia_2.txt'
SIZE = 32

def read_labeled_image_list(image_list_file):
    """
    Reading labeled images from a list
    :param image_list_file: the path of the file
    :return: filenames and labels of the dataset
    """
    with open(image_list_file, 'r') as f:
        print('Opened image list file')
        image_list = []
        label_list = []
        for idx, line in enumerate(f):
            filename, label = line[:-1].split(' ')[:2]
            label = int(label)
            if os.path.exists('data/' + filename):
                if idx % 100000 == 0:
                    print('Inputted %d lines' % idx)
                if label < SIZE:
                    image_list.append(filename)
                    label_list.append(int(label))
            else:
                print('File not found: ' + filename)
        print('Return list.')
    return image_list, label_list


def output_image_list(output_file, image_list, label_list):
    with open(output_file, 'w') as f:
        for image, label in zip(image_list, label_list):
            f.write(image + ' ' + str(label) + '\n')


if __name__ == "__main__":
    image_list, label_list = read_labeled_image_list(INPUT)
    print(image_list)
    print(label_list)
    output_image_list(OUTPUT, image_list, label_list)
