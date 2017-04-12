import sys
import cv2
import numpy as np
import math
from random import shuffle
import configparser
if (sys.version == 2):
    import cPickle
else:
    import _pickle as cPickle

def load_training_data(data_file_name):
    training_data = None
    with open(data_file_name, 'rb') as f:
        training_data = cPickle.load(f, encoding='iso-8859-1')

    index = [i for i in  range(len(training_data[0]))]
    shuffle(index)

    return training_data[0][index],training_data[1][index]

def getmask(patch_size = 21):
    r = patch_size / 2
    center = ((patch_size + 1) / 2, (patch_size + 1) / 2)
    min_mask = np.zeros((patch_size, patch_size))
    mask = np.zeros((patch_size, patch_size))
    inner_mask = np.zeros((patch_size, patch_size))
    for x in range(patch_size):
        for y in range(patch_size):
            if distanse((x + 1, y + 1), center) <= r:
                mask[x, y] = 1.0
            if distanse((x + 1, y + 1), center) <= r - 2:
                inner_mask[x, y] = 1.0
            if distanse((x + 1, y + 1), center) <= r - 4:
                min_mask[x, y] = 1.0

    return inner_mask

def psnr(target, ref, scale = 3):
    # assume RGB image
    target_data = np.array(target)
    target_data = target_data[scale:-scale, scale:-scale]

    ref_data = np.array(ref)
    ref_data = ref_data[scale:-scale, scale:-scale]

    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(1.0 / rmse)


def ssim(i1, i2):

    C1 = 6.5025
    C2 = 58.5225

    I1 = i1 * 1.0
    I2 = i2 * 1.0

    I1_2 = I1 * I1

    I2_2 = I2 * I2

    I1_I2 = I1 * I2

    mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)

    mu1_2 = mu1 * mu1

    mu2_2 = mu2 * mu2

    mu1_mu2 = mu1 * mu2

    sigma1_2 = cv2.GaussianBlur(I1_2, (11, 11), 1.5)
    sigma1_2 = sigma1_2 -mu1_2

    sigam2_2 = cv2.GaussianBlur(I2_2, (11, 11), 1.5)
    sigam2_2 -= mu2_2

    sigam12 = cv2.GaussianBlur(I1_I2, (11, 11), 1.5)
    sigam12 -= mu1_mu2

    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigam12 + C2
    t3 = t1 * t2

    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigam2_2 + C2
    t1 = t1 * t2

    ssim_map = t3 / t1
    ssim = np.mean(ssim_map)

    return ssim

def toYchannel(RGBimg):
    img_ycrcb_h = cv2.cvtColor(np.array(RGBimg/ 255.0 , dtype=np.float32), cv2.COLOR_BGR2YCR_CB)
    return img_ycrcb_h[:,:,0]


def distanse(p1, p2):
    return np.sqrt(np.power(p1[0] - p2[0],2) + np.power(p1[1] - p2[1],2))

def patch_dist(p1, p2):
    return np.sqrt(np.mean(np.power(p1 - p2,2)))
