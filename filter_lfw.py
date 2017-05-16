#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/5/16 下午3:09
# @Author  : irmo

file = 'pts.txt'
output = 'lfw_landmark.txt'


def check_line(line):
    if 'lfw' in line:
        return True


def main():
    f = open(file, 'r')
    out = open(output, 'w')
    while True:
        line = f.readline()
        if not line:
            break
        if check_line(line):
            bounding_box = f.readline().split()
            l1 = f.readline().split()
            l2 = f.readline().split()
            l3 = f.readline().split()
            l4 = f.readline().split()
            l5 = f.readline().split()
            filename = '/'.join(line.split('/')[-2:]).strip()
            new_line_list = [filename] + bounding_box + l1 + l2 + l3 + l4 + l5
            new_line = ' '.join(new_line_list)
            out.write(new_line + '\n')


if __name__ == "__main__":
    main()
