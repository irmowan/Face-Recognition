from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.PIXEL_MEANS = 0

__C.scale = 255

__C.crop_size = 128

# Resize mode: resize the rotated image according to the distance between
# 0: two eyes.
# 1: center of two eyes and mouth corner.
__C.resize_mode = 1

#Crop mode: Crop face area mode.
# 1: Eye center
# 2: Left eye
# 3: Right eye
# 4: Nose
# 5: Left mouth corner
# 6: Right mouth corner
# 7: Random
__C.crop_mode = 1

__C.ec_mc_y = 48

__C.ec_y = 40

__C.eye_dist = 48

__C.forcegray = True
