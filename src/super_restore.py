# coding:utf-8
import cv2
from scipy.misc import imresize
import time
import os
from data_utils import *

class Retore(object):
    def __init__(self):
        self.input_holder = ""
        self.sess =""
        self.mid_res = {}
        self.model = None

    def init_param(self, model, over_lap=0,scale = 3.0):
        self.model = model
        self.scale = scale
        self.over_lap = over_lap

    def super_resolution_image(self, img_h):
        tr = self.model
        patch_size = tr.input_size
        over_lap = self.over_lap
        scale = self.scale

        img_low = imresize(img_h, 1 / scale, interp='bicubic')
        img_l = imresize(img_low, scale, interp='bicubic')

        img_ycrcb_h = cv2.cvtColor(np.array(img_h / 255.0,dtype=np.float32), cv2.COLOR_BGR2YCR_CB)
        img_ycrcb_l = cv2.cvtColor(np.array(img_l / 255.0,dtype=np.float32), cv2.COLOR_BGR2YCR_CB)

        hr_bicubic_y_channel = img_ycrcb_l[:, :, 0]

        size = hr_bicubic_y_channel.shape
        restored_residual = np.zeros_like(hr_bicubic_y_channel)
        restored_count = np.ones((size[0], size[1])) * 1e-5
        one_patch = np.ones((patch_size, patch_size))

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
                lr_patch_list.append(hr_bicubic_y_channel[x:x + patch_size, y:y + patch_size])
                # 这个数据只有在训练的时候才有用，所以这里没有用处
                residual_patch_list.append(hr_bicubic_y_channel[x:x + patch_size, y:y + patch_size])

        tmp_len = len(lr_patch_list)
        while len(lr_patch_list) % tr.batch_size!=0:
            lr_patch_list.append(lr_patch_list[0])
            residual_patch_list.append(residual_patch_list[0])
        tr.set_test_data(np.array(lr_patch_list), np.array(residual_patch_list))
        restore_patch_list = tr.restoring()[0:tmp_len]
        restore_patch_list = [np.reshape(patch,(patch_size, patch_size)) for patch in restore_patch_list]

        i = 0
        for x in xgrid:
            for y in ygrid:
                # restored_residual[x:x + patch_size, y:y + patch_size] = restore_patch_list[i] + restored_residual[x:x + patch_size, y:y + patch_size]
                restored_residual[x:x + patch_size, y:y + patch_size] = restore_patch_list[i] + restored_residual[x:x + patch_size, y:y + patch_size]
                i = i + 1
                restored_count[x:x + patch_size, y:y + patch_size] = restored_count[x:x + patch_size, y:y + patch_size] + 1.0

        img_ycrcb_l[:, :, 0] = hr_bicubic_y_channel + restored_residual/restored_count

        img_restored = cv2.cvtColor(img_ycrcb_l, cv2.COLOR_YCR_CB2BGR, None)
        # print("灰度图像的psnr%.2f" % psnr(img_ycrcb_l[:, :, 0], img_ycrcb_h[:, :, 0]))
        return img_restored * 255

    def super_resolution_image2(self, img_h):
        tr = self.model
        patch_size = tr.input_size
        over_lap = self.over_lap
        scale = self.scale

        img_low = imresize(img_h, 1 / scale, interp='bicubic')
        img_l = imresize(img_low, scale, interp='bicubic')

        img_ycrcb_h = cv2.cvtColor(np.array(img_h / 255.0,dtype=np.float32), cv2.COLOR_BGR2YCR_CB)
        img_ycrcb_l = cv2.cvtColor(np.array(img_l / 255.0,dtype=np.float32), cv2.COLOR_BGR2YCR_CB)

        hr_bicubic_y_channel = img_ycrcb_l[:, :, 0]

        size = hr_bicubic_y_channel.shape


        img_restored = cv2.cvtColor(img_ycrcb_l, cv2.COLOR_YCR_CB2BGR, None)
        # print("灰度图像的psnr%.2f" % psnr(img_ycrcb_l[:, :, 0], img_ycrcb_h[:, :, 0]))
        return img_restored * 255

def restore_dir(dir, tr, store_path=""):
    if store_path == "":
        store_path="tmp"
    file_list = os.listdir(dir)
    result_dir = "%s/%s/" % (dir,store_path)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    r = Retore()
    r.init_param(tr,scale = 3.0,over_lap=0)
    psnr_sum = 0.0
    ssim_sum = 0.0
    time_sum = 0.0
    image_num = 0
    # 遍历文件目录
    for file_name in file_list:
        if file_name[-3:] != "bmp" and file_name[-3:] != "jpg"  and file_name[-3:] != "png":
            continue
        print("当前处理文件%s" % file_name)
        img = cv2.imread(dir + "/" + file_name)

        img = img[0: img.shape[0] - img.shape[0] % 3, 0: img.shape[1] - img.shape[1] % 3, :]

        start_time = time.time()
        img_restored = r.super_resolution_image(img)

        # 结果评价
        time_cost = time.time() - start_time
        tmp_psnr = psnr(img/255.0 , img_restored/255.0)
        tmp_ssim = ssim(img, img_restored)
        print("RGB图像的psnr:%.2f, ssim:%.2f, time:%.2f" % (tmp_psnr,tmp_ssim,time_cost))
        cv2.imwrite(result_dir + file_name, img_restored)
        psnr_sum = psnr_sum + tmp_psnr
        time_sum = time_sum + time_cost
        ssim_sum = ssim_sum + tmp_ssim
        image_num = image_num + 1.0
    print("*************************************************************")
    print("平均psnr:%.2f, ssim:%.2f,time:%.2f" % (psnr_sum / image_num, ssim_sum / image_num, time_sum / image_num))
    print("*************************************************************")


