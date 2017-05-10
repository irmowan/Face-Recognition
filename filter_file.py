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
        line_list = []
        for idx, line in enumerate(f):
            filename, label = line[:-1].split(' ')[:2]
            label = int(label)
            if os.path.exists('data/' + filename):
                if idx % 100000 == 0:
                    print('Inputted %d lines' % idx)
                if label < SIZE:
                    line_list.append(line)
            else:
                print('File not found: ' + filename)
        print('Return list.')
    return line_list 


def output_image_list(output_file, line_list):
    with open(output_file, 'w') as f:
        for line in line_list:
            f.write(line)


if __name__ == "__main__":
    line_list = read_labeled_image_list(INPUT)
    print(line_list)
    output_image_list(OUTPUT, line_list)
