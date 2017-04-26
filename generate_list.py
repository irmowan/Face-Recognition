#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/4/26 下午5:25
# @Author  : irmo

import os
import os.path

OUTPUT_FILE = "list.txt"
root_dir = "data/"

if __name__ == "__main__":
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w") as f:
            for _0, dir_names, _1 in os.walk(root_dir):
                for label in dir_names:
                    for _2, _3, images in os.walk(root_dir + label):
                        for image in images:
                            if image[-3:] == 'jpg':
                                f.write(label + '/' + image + ' ' + label + '\n')
