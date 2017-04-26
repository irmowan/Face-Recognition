#! /usr/bin/python
# -*- coding: utf8 -*-


def validate(line):
    """
    :param line: check whether the line is valid
    :return:
    """
    try:
        img_path, label, bb, landmark = line.split()
        return img_path, label, bb, landmark
    except Exception as e:
        print(e)
        return None


def align(img_path, label, bb, landmark):
    """
    :param img_path: path of the input image
    :param label: label of the image
    :param bb: bounding box
    :param landmark: landmark points
    :return: 128*128 aligned picture
    """
    pass


def load_list(list_path):
    """
    Reading the list txt, validate each line and align the picture
    :param list_path: the path of the list txt file
    :return: none
    """
    f = open(list_path)
    for line in f:
        try:
            img_path, label, bb, landmark = validate(line)
            pic = align(img_path, label, bb, landmark)
            return pic
        except Exception as e:
            print(e)
            return None


if __name__ == "__main__":
    pass
