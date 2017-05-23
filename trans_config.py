class Config():
    def __init__(self):
        self.PIXEL_MEANS = 0
        self.scale = 255
        self.crop_size = 224 
        # Resize mode: resize the rotated image according to the distance between
        # 0: two eyes.
        # 1: center of two eyes and mouth corner.
        self.resize_mode = 1
        # Crop mode: Crop face area mode.
        # 1: Eye center
        # 2: Left eye
        # 3: Right eye
        # 4: Nose
        # 5: Left mouth corner
        # 6: Right mouth corner
        # 7: Random
        self.crop_mode = 1
        self.ec_mc_y = 84
        self.ec_y = 70
        self.eye_dist = 84
        self.forcegray = False 


cfg = Config()
