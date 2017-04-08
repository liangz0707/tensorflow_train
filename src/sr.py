import numpy as np
import cv2
from super_trainer import *
import matplotlib.pyplot as plt
import math
from scipy.misc import imresize

def toYchannel(RGBimg):
    img_ycrcb_h = cv2.cvtColor(np.array(RGBimg/ 255.0 , dtype=np.float32), cv2.COLOR_BGR2YCR_CB)
    return img_ycrcb_h[:,:,0]


def psnr(target, ref, scale):
    # assume RGB image
    target_data = np.array(target)
    target_data = target_data[scale:-scale, scale:-scale]

    ref_data = np.array(ref)
    ref_data = ref_data[scale:-scale, scale:-scale]

    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(1.0 / rmse)


class Retore(object):
    def __init__(self):
        self.input_holder = ""
        self.sess =""
        self.mid_res = {}


    def super_resolution_image(self, img_h, tr, scale=3.0, patch_size =21, over_lap=10):
        img_low = imresize(img_h, 1 / 3.0, interp='bicubic')
        img_l = imresize(img_low, 3.0, interp='bicubic')

        img_ycrcb_h = cv2.cvtColor(np.array(img_h / 255.0,dtype=np.float32), cv2.COLOR_BGR2YCR_CB)
        img_ycrcb_l = cv2.cvtColor(np.array(img_l / 255.0,dtype=np.float32), cv2.COLOR_BGR2YCR_CB)

        hr_bicubic_y_channel = img_ycrcb_l[:, :, 0]

        size = hr_bicubic_y_channel.shape
        restored_residual = np.zeros_like(hr_bicubic_y_channel)
        restored_count = np.ones((size[0], size[1])) * 1e-5

        xgrid = np.ogrid[0:size[0] - patch_size: patch_size - over_lap]
        ygrid = np.ogrid[0:size[1] - patch_size: patch_size - over_lap]

        # 将image转换成patch
        lr_patch_list = []
        restore_patch_list = None
        residual_patch_list = []
        lr_mean_list = []
        for x in xgrid:
            for y in ygrid:
                m = np.mean(hr_bicubic_y_channel[x:x + patch_size, y:y + patch_size])
                lr_mean_list.append(m)
                lr_patch_list.append(hr_bicubic_y_channel[x:x + patch_size, y:y + patch_size] - m)
                #lr_patch_list.append(hr_bicubic_y_channel[x:x + patch_size, y:y + patch_size])
                # 这个数据只有在训练的时候才有用，所以这里没有用处
                residual_patch_list.append(hr_bicubic_y_channel[x:x + patch_size, y:y + patch_size])

        tmp_len = len(lr_patch_list)
        while len(lr_patch_list) % tr.batch_size!=0:
            lr_patch_list.append(lr_patch_list[0])
            residual_patch_list.append(residual_patch_list[0])
        tr.set_data(np.array(lr_patch_list), np.array(residual_patch_list))
        restore_patch_list = tr.restoring()[0:tmp_len]
        restore_patch_list = [np.reshape(patch,(21,21)) for patch in restore_patch_list]

        i = 0
        for x in xgrid:
            for y in ygrid:
                # restored_residual[x:x + patch_size, y:y + patch_size] = restore_patch_list[i] + restored_residual[x:x + patch_size, y:y + patch_size]
                restored_residual[x:x + patch_size, y:y + patch_size] = restore_patch_list[i] + restored_residual[x:x + patch_size, y:y + patch_size]* 1.3
                i = i + 1
                restored_count[x:x + patch_size, y:y + patch_size] = restored_count[x:x + patch_size, y:y + patch_size] + 1.0

        img_ycrcb_l[:, :, 0] = hr_bicubic_y_channel + restored_residual/restored_count

        img_restored = cv2.cvtColor(img_ycrcb_l, cv2.COLOR_YCR_CB2BGR, None)
        print("灰度图像的psnr%.2f" % psnr(img_ycrcb_l[:, :, 0], img_ycrcb_h[:, :, 0]))
        return img_restored * 255

# def psnr(img1,img2, m = 1.17):
#     img2 = 1.0* img2/ m
#     img1 = 1.0* img1/ m
#     s = img1.shape
#     si = 1
#     if len(s) ==3:
#         si = s[0] * s[1] * s[2]
#     if len(s) ==2:
#         si = s[0] * s[1]
#     E = img1 - img2
#     mse = np.sum(np.power(E,2.0) )
#     return 10.0 * math.log10( si * 1.0 / mse)


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

import os
def restore_dir(dir):
    file_list = os.listdir(dir)
    result_dir = dir +"/"+ "cnn_restore_decon3/"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    r = Retore()
    psnr_sum = 0.0
    image_num = 0
    with tf.device("/gpu:0"):
        with tf.Graph().as_default():
            tr = SRTrainerDecon3(model_save_file="E:/mySuperResolution/dataset/test_291",
                            model_load_file="E:/mySuperResolution/dataset/Y_291_decon_v3-0",
                            model_tag=0)
            # tr = SRTrainer(model_save_file="E:/mySuperResolution/dataset/test_291",
            #                    model_load_file="E:/mySuperResolution/dataset/Y_291_decon_v2-0",
            #                    model_tag=0)
            tr.init_param(batch_size=32,layer_num=8, input_size=21)
            tr.setup_frame()
            for file_name in file_list:
                if file_name[-3:] != "bmp" and file_name[-3:] != "jpg":
                    continue
                print(file_name)
                img = cv2.imread(dir + "/" + file_name)

                img = img[0: img.shape[0] - img.shape[0] % 3, 0: img.shape[1] - img.shape[1] % 3, :]
                img_restored = r.super_resolution_image(img, tr)
                tmp_psnr = psnr(img/255.0 , img_restored/255.0)
                print("RGB图像的psnr%.2f" % tmp_psnr)
                # 进行保存和统计
                cv2.imwrite(result_dir + file_name, img_restored)
                psnr_sum = psnr_sum + tmp_psnr
                image_num = image_num + 1.0

            print(psnr_sum / image_num)

if __name__ == '__main__':
    # a = cv2.imread("E:\\mySuperResolution\\dataset\\Set14\\results_Set14_x3_1024atoms\\monarch[1-Original].bmp")
    # a = a[:a.shape[0] - a.shape[0]%3, :a.shape[1] - a.shape[1]%3]
    # b = cv2.imread("E:\\mySuperResolution\\dataset\\Set14\\results_Set14_x3_1024atoms\\monarch[12-JOR].bmp")
    # c = cv2.imread("E:\\mySuperResolution\\dataset\\Set14\\cnn_restore\\monarch[1-Original].bmp")
    # d = cv2.imread("E:\\mySuperResolution\\dataset\\Set14\\results_Set14_x3_1024atoms\\monarch[2-Bicubic].bmp")
    #
    #
    # # 使用scipy的bicubic线性插值的结果
    # print(" 使用scipy的bicubic线性插值的结果")
    # a_l = imresize(a,1/3.0, interp='bicubic')
    # a_r = imresize(a_l,  3.0, interp='bicubic')
    # print (psnr(a/255.0,a_r*1.0/255.0))
    # print(psnr(toYchannel(a), toYchannel(a_r)))
    #
    # # 使用opencv的bicubic线性插值的结果
    # print(" 使用opencv的bicubic线性插值的结果")
    # a_l = cv2.resize(a, (0, 0), None, 1.0 / 3, 1.0 / 3)
    # a_r = cv2.resize(a_l, (0, 0), None, 3, 3, interpolation=cv2.INTER_CUBIC)
    # print(psnr(a/255.0, a_r/255.0))
    # print(psnr(toYchannel(a), toYchannel(a_r)))
    #
    # print("JOR")
    # print(psnr(a[:504,:762,:]/255.0 ,b/255.0))
    # print(psnr(toYchannel(a), toYchannel(b)))
    # # 在lab空间用opencv  bicubic插值的结果
    # print("在Ycbcr空间用opencv cnn插值的结果")
    # print(psnr(a/255.0 ,c/255.0))
    # print(psnr(toYchannel(a), toYchannel(c)))
    #
    # print("在Bicubic的结果")
    # print(psnr(d/255.0 ,a*1.0/255.0 ))
    # print(psnr(toYchannel(a), toYchannel(d)))

    restore_dir("E:\\mySuperResolution\\dataset\\Set5")
