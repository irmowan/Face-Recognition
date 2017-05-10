from __future__ import absolute_import
from __future__ import division

import numpy as np
import cv2, math

from trans_config import cfg


def point_trans(ori_point, angle, ori_shape, new_shape):
    """ Transfrom the point from original to rotated image.
    Args:
        ori_point: Point coordinates in original image.
        angle: Rotate angle.
        ori_shape: The shape of original image.
        new_shape: The shape of rotated image.

    Returns:
        Numpy array of new point coordinates in rotated image.
    """
    dx = ori_point[0] - ori_shape[1] / 2.0
    dy = ori_point[1] - ori_shape[0] / 2.0

    t_x = round(dx * math.cos(angle) - dy * math.sin(angle) + new_shape[1] / 2.0)
    t_y = round(dx * math.sin(angle) + dy * math.cos(angle) + new_shape[0] / 2.0)
    return np.array((int(t_x), int(t_y)))


def im_rotate(im, landmark):
    """Rotate the image according to the angle of two eyes.

    Args:
        landmark: 5 points, left_eye, right_eye, nose, leftmouth, right_mouth
        im: image matrix

    Returns:
        A rotated image matrix.
        Rotated angle.
        Rotated landmark points.
    """
    ang = math.atan2(landmark[3] - landmark[1], landmark[2] - landmark[0])
    angle = ang / math.pi * 180
    center = tuple(np.array((im.shape[1] / 2.0, im.shape[0] / 2.0)))
    scale = 1.0

    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    dst = cv2.warpAffine(im, rot_mat, (im.shape[1], im.shape[0]))
    # rotate 5 landmark points
    left_eye = point_trans(landmark[0:2], -ang, im.shape, im.shape)
    right_eye = point_trans(landmark[2:4], -ang, im.shape, im.shape)
    nose = point_trans(landmark[4:6], -ang, im.shape, im.shape)
    left_mouth = point_trans(landmark[6:8], -ang, im.shape, im.shape)
    right_mouth = point_trans(landmark[8:10], -ang, im.shape, im.shape)
    n_landmark = np.concatenate([left_eye, right_eye, nose, left_mouth, right_mouth])
    return dst, ang, n_landmark


def im_resize(im, landmark, ang):
    """Resize the image according to the distance between eyes or mouth.

    Args:
        rot_landmark: rotated 5 landmark points
        im: rotated image
        mode: resize mode

    Return:
        A resized image.
        Resize scale.
        resized landmark points.
    """
    if cfg.resize_mode == 0:
        resize_scale = float(cfg.eye_dist) / float(landmark[2] - landmark[0])
        resize_x = int(round(im.shape[1] * resize_scale))
        resize_y = int(round(im.shape[0] * resize_scale))
        im_resize = cv2.resize(im, (resize_x, resize_y),
                               interpolation=cv2.INTER_LINEAR)
        rez_landmark = np.round(landmark.astype(np.float) * resize_scale).astype(np.int)
        # return im_resize, resize_scale, rez_landmark
    elif cfg.resize_mode == 1:
        eye_c_y = int(round((landmark[1] + landmark[3]) / 2.0))
        mouth_c_y = int(round((landmark[7] + landmark[9]) / 2.0))
        resize_scale = float(cfg.ec_mc_y) / float(mouth_c_y - eye_c_y)
        resize_x = int(round(im.shape[1] * resize_scale))
        resize_y = int(round(im.shape[0] * resize_scale))
        im_resize = cv2.resize(im, (resize_x, resize_y),
                               interpolation=cv2.INTER_LINEAR)
        rez_landmark = np.round(landmark.astype(np.float) * resize_scale).astype(np.int)
    return im_resize, resize_scale, rez_landmark


def im_crop(im, landmark, resize_scale):
    """Crop resized image according to the bounding box or landmark points.

    Args:
        landmark: rotated and resized point transformed landmark.
        im: images after rotation and resize.
        resize_scale: resize scale from rotated image.
        ori_shape: the rotated image size

    Returns:
        Cropped image, pad cropped image to the crop size.
    """
    if cfg.crop_mode == 7:
        crop_mode = random.randint(1, 6)
    else:
        crop_mode = cfg.crop_mode

    anchor_x = 0
    anchor_y = 0
    dy = 0
    crop = np.zeros((cfg.crop_size, cfg.crop_size, im.shape[2]), dtype=np.uint8)
    if crop_mode == 1:
        eye_c = (landmark[0:2] + landmark[2:4]) / 2
        anchor_x = eye_c[0]
        anchor_y = eye_c[1]
        dy = anchor_y - cfg.ec_y
    elif crop_mode == 2:
        anchor_x = landmark[0]
        anchor_y = landmark[1]
        dy = anchor_y - int(round(cfg.crop_size / 2))
    elif crop_mode == 3:
        anchor_x = landmark[2]
        anchor_y = landmark[3]
        dy = anchor_y - int(round(cfg.crop_size / 2))
    elif crop_mode == 4:
        anchor_x = landmark[4]
        anchor_y = landmark[5]
        dy = anchor_y - int(round(cfg.crop_size / 2))
    elif crop_mode == 5:
        anchor_x = landmark[6]
        anchor_y = landmark[7]
        dy = anchor_y - int(round(cfg.crop_size / 2))
    elif crop_mode == 6:
        anchor_x = landmark[8]
        anchor_y = landmark[9]
        dy = anchor_y - int(round(cfg.crop_size / 2))

    crop_x = anchor_x - int(round(cfg.crop_size / 2))
    crop_x_end = crop_x + cfg.crop_size - 1
    crop_y = dy
    crop_y_end = crop_y + cfg.crop_size - 1
    box_x = guard(np.array([crop_x, crop_x_end]), im.shape[0])
    box_y = guard(np.array([crop_y, crop_y_end]), im.shape[1])
    crop[(box_y[0] - crop_y):(box_y[1] - crop_y + 1), (box_x[0] - crop_x):(box_x[1] - crop_x + 1), :] = \
        im[box_y[0]:box_y[1] + 1, box_x[0]:box_x[1] + 1, :]
    return crop


def guard(x, N):
    if x[0] < 0:
        x[0] = 0
    if x[1] > N - 1:
        x[1] = N - 1
    return x


def img_process(im, landmark):
    """
    Image processing, rotate, resize, and crop the face image.

    Args:
        im: numpy array, Original image
        landmark: 5 landmark points

    Return:
        Crop face region
    """
    im_rot, ang, r_landmark = im_rotate(im, landmark)
    im_rez, resize_scale, rez_landmark = im_resize(im_rot, r_landmark, ang)
    crop = im_crop(im_rez, rez_landmark, resize_scale)
    if cfg.forcegray == True:
        crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return (crop.astype(np.float) - cfg.PIXEL_MEANS) / cfg.scale
