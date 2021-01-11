import cv2
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
import glob
from fast_poisson import fast_poisson
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from skimage.restoration import inpaint
import pdb

class image_processor:
    def __init__(self):
        pass
    
    def crop_image(self,img, pad):
        return img[pad:-pad,pad:-pad]

class calibration:
    def __init__(self):
        self.BallRad= 4.76/2 #4.76/2 #mm
        self.Pixmm =  .10577 #4.76/100 #0.0806 * 1.5 mm/pixel
        self.ratio = 1/2.
        self.red_range = [-90, 90]
        self.green_range = [-90, 90] #[-60, 50]
        self.blue_range = [-90, 90] # [-80, 60]
        self.red_bin = int((self.red_range[1] - self.red_range[0])*self.ratio)
        self.green_bin = int((self.green_range[1] - self.green_range[0])*self.ratio)
        self.blue_bin = int((self.blue_range[1] - self.blue_range[0])*self.ratio)
        self.zeropoint = -90
        self.lookscale = 180
        self.bin_num = 90
    
    def mask_marker(self, raw_image):
        m, n = raw_image.shape[1], raw_image.shape[0]
        raw_image = cv2.pyrDown(raw_image).astype(np.float32)
        blur = cv2.GaussianBlur(raw_image, (25, 25), 0)
        blur2 = cv2.GaussianBlur(raw_image, (5, 5), 0)
        diff = blur - blur2
        diff *= 16.0
        # cv2.imshow('blur2', diff.astype(np.uint8))
        # cv2.waitKey(10000)
        

        diff[diff < 0.] = 0.
        diff[diff > 255.] = 255.

        # diff = cv2.GaussianBlur(diff, (5, 5), 0)
        # cv2.imshow('diff', diff.astype(np.uint8))
        # cv2.waitKey(10000)
        # pdb.set_trace()
        mask = (diff[:, :, 0] > 25) & (diff[:, :, 2] > 25) & (diff[:, :, 1] >
                                                              120)
        # cv2.imshow('mask', mask.astype(np.uint8) * 255)
        # cv2.waitKey(1000)

        # pdb.set_trace()
        mask = cv2.resize(mask.astype(np.uint8), (m, n))
#        mask = mask * self.dmask
#        mask = cv2.dilate(mask, self.kernal4, iterations=1)

        # mask = cv2.erode(mask, self.kernal4, iterations=1)
        # cv2.imshow('mask', a.astype(np.uint8) * 255)
        # cv2.waitKey(100000)
        return (1 - mask) * 255



if __name__ == '__main__':
    cali = calibration()
    imp = image_processor()
    pad = 20
    ref_img = cv2.imread('./test_data/ref.jpg')
    ref_img = imp.crop_image(ref_img, pad)
    marker = cali.mask_marker(ref_img)

