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

    def init_param(self, model, over_lap=0,scale = 3.0, patch_size=21):
        self.model = model
        self.scale = scale
        self.over_lap = over_lap

        self.patch_size = patch_size
        mask = np.zeros((patch_size, patch_size))
        inner_mask = np.zeros((patch_size, patch_size))
        min_mask = np.zeros((patch_size, patch_size))
        center = ((patch_size + 1) / 2, (patch_size + 1) / 2)
        self.center = center
        r = patch_size / 2

        gaussianKernel = cv2.getGaussianKernel(self.patch_size, 3)
        self.gaussianKernel2D = gaussianKernel * gaussianKernel.T

        for x in range(patch_size):
            for y in range(patch_size):
                if distanse((x + 1, y + 1), center) <= r:
                    mask[x, y] = 1.0
                if distanse((x + 1, y + 1), center) <= r - 2:
                    inner_mask[x, y] = 1.0
                if distanse((x + 1, y + 1), center) <= r - 4:
                    min_mask[x, y] = 1.0
        self.mask = mask
        self.min_mask = min_mask
        # 比正常的mask小一圈，为了消除在旋转当中产生的一些边界像素的误差
        self.inner_mask = inner_mask
        self.degree_mat_list = []
        for i in range(0,360):
            M = cv2.getRotationMatrix2D(self.center, i, 1)
            self.degree_mat_list.append(M)

        self.degree_rmat_list = []
        for i in range(0,360):
            M = cv2.getRotationMatrix2D(self.center, -i, 1)
            self.degree_rmat_list.append(M)

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
        hr_GT_y_channel = img_ycrcb_h[:, :, 0]

        size = hr_bicubic_y_channel.shape
        restored_residual = np.zeros_like(hr_bicubic_y_channel)
        restored_count = np.ones((size[0], size[1])) * 1e-5
        one_patch = np.ones((patch_size, patch_size))

        xgrid = np.ogrid[0:size[0] - patch_size: patch_size - over_lap]
        ygrid = np.ogrid[0:size[1] - patch_size: patch_size - over_lap]

        # 将image转换成patch
        lr_patch_list = []
        restore_patch_list = []
        residual_patch_list = []
        for x in xgrid:
            for y in ygrid:
                lr_patch_list.append(hr_bicubic_y_channel[x:x + patch_size, y:y + patch_size])
                # 这个数据只有在训练的时候才有用，所以这里没有用处
                residual_patch_list.append(hr_GT_y_channel[x:x + patch_size, y:y + patch_size] - hr_bicubic_y_channel[x:x + patch_size, y:y + patch_size])

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
                restored_residual[x:x + patch_size, y:y + patch_size] = 0.85*restore_patch_list[i] + 0.15*residual_patch_list[i] + restored_residual[x:x + patch_size, y:y + patch_size]
                i = i + 1
                restored_count[x:x + patch_size, y:y + patch_size] = restored_count[x:x + patch_size, y:y + patch_size] + 1.0

        img_ycrcb_l[:, :, 0] = hr_bicubic_y_channel + restored_residual/restored_count

        img_restored = cv2.cvtColor(img_ycrcb_l, cv2.COLOR_YCR_CB2BGR, None)
        # print("灰度图像的psnr%.2f" % psnr(img_ycrcb_l[:, :, 0], img_ycrcb_h[:, :, 0]))

        # n = 0
        # p = residual_patch_list[0] * 0
        # for (a,b) in zip(restore_patch_list, residual_patch_list):
        #     p = p + np.power(a-b, 2)
        #     n = n + 1
        # plt.imshow(p/n)
        # plt.show()
        return img_restored * 255

    def super_resolution_image_rotate(self, img_h):
        tr = self.model
        patch_size = tr.input_size
        over_lap = self.over_lap
        scale = self.scale

        img_low = imresize(img_h, 1 / scale, interp='bicubic')
        img_l = imresize(img_low, scale, interp='bicubic')

        img_ycrcb_h = cv2.cvtColor(np.array(img_h / 255.0, dtype=np.float32), cv2.COLOR_BGR2YCR_CB)
        img_ycrcb_l = cv2.cvtColor(np.array(img_l / 255.0, dtype=np.float32), cv2.COLOR_BGR2YCR_CB)
        hr_bicubic_y_channel = img_ycrcb_l[:, :, 0]
        hr_GT_y_channel = img_ycrcb_h[:, :, 0]

        size = hr_bicubic_y_channel.shape
        restored_residual = np.zeros_like(hr_bicubic_y_channel)
        restored_count = np.ones((size[0], size[1])) * 1e-5
        one_patch = np.ones((patch_size, patch_size))

        xgrid = np.ogrid[0:size[0] - patch_size: patch_size - over_lap]
        ygrid = np.ogrid[0:size[1] - patch_size: patch_size - over_lap]

        # 将image转换成patch
        lr_patch_list = []
        restore_patch_list = []
        residual_patch_list = []
        for x in xgrid:
            for y in ygrid:
                lr_patch_list.append(hr_bicubic_y_channel[x:x + patch_size, y:y + patch_size])
                # 这个数据只有在训练的时候才有用，所以这里没有用处
                residual_patch_list.append(
                    hr_GT_y_channel[x:x + patch_size, y:y + patch_size] - hr_bicubic_y_channel[x:x + patch_size,
                                                                          y:y + patch_size])
        final_restore_patch_list = [p for p in residual_patch_list]
        dist = [100000 for p in residual_patch_list]
        tmp_len = len(lr_patch_list)
        while len(lr_patch_list) % tr.batch_size != 0:
            lr_patch_list.append(lr_patch_list[0])
            residual_patch_list.append(residual_patch_list[0])


        for i in range(360):
            tmp_lr_patch_list   = [cv2.warpAffine(p, self.degree_mat_list[i], (self.patch_size, self.patch_size)) * self.inner_mask for p in lr_patch_list]
            tmp_residual_patch_list = [cv2.warpAffine(p, self.degree_mat_list[i], (self.patch_size, self.patch_size)) * self.inner_mask for p in residual_patch_list]

            tr.set_test_data(np.array(tmp_lr_patch_list), np.array(tmp_residual_patch_list))
            restore_patch_list = tr.restoring()[0:tmp_len]
            for index, ds in enumerate(dist):
                new_ds = patch_dist(restore_patch_list[index][:,:,0] * self.inner_mask, tmp_residual_patch_list[index])

                if ds > new_ds:
                    dist[index] = new_ds
                    final_restore_patch_list[index] = cv2.warpAffine(restore_patch_list[index][:,:,0], self.degree_rmat_list[i], (self.patch_size, self.patch_size)) * self.inner_mask
        #
        # tr.set_test_data(np.array(lr_patch_list), np.array(residual_patch_list))
        # final_restore_patch_list = tr.restoring()[0:tmp_len]

        restore_patch_list = [np.reshape(patch, (patch_size, patch_size)) for patch in final_restore_patch_list]

        i = 0
        for x in xgrid:
            for y in ygrid:
                restored_residual[x:x + patch_size, y:y + patch_size] = self.inner_mask * restore_patch_list[i] + restored_residual[
                                                                                                x:x + patch_size,
                                                                                                y:y + patch_size]
                i = i + 1
                restored_count[x:x + patch_size, y:y + patch_size] = restored_count[x:x + patch_size,
                                                                     y:y + patch_size] + one_patch * self.inner_mask

        img_ycrcb_l[:, :, 0] = hr_bicubic_y_channel + restored_residual / restored_count

        img_restored = cv2.cvtColor(img_ycrcb_l, cv2.COLOR_YCR_CB2BGR, None)
        # print("灰度图像的psnr%.2f" % psnr(img_ycrcb_l[:, :, 0], img_ycrcb_h[:, :, 0]))
        # n = 0
        # p = residual_patch_list[0] * 0
        # for (a,b) in zip(restore_patch_list, residual_patch_list):
        #     p = p + np.power(a-b, 2)
        #     n = n + 1
        # plt.imshow(p/n)
        # plt.show()
        return img_restored * 255

def restore_dir(dir, tr, store_path=""):
    if store_path == "":
        store_path="tmp"
    file_list = os.listdir(dir)
    result_dir = "%s/%s/" % (dir,store_path)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    r = Retore()
    r.init_param(tr,scale = 3.0,over_lap=10)
    psnr_sum = 0.0
    ssim_sum = 0.0
    time_sum = 0.0
    image_num = 0
    # 遍历文件目录
    error_list = []
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

        img_restored[img_restored>255] = 255
        img_restored[img_restored<0] = 0
        #tmp_psnr = psnr(img/255.0 , img_restored/255.0)
        tmp_psnr = psnrRGB(img , img_restored)
        #tmp_ssim = ssim(img, img_restored)
        tmp_ssim = ssimRGB(img , img_restored)
        print("RGB图像的psnr:%.2f, ssim:%.2f, time:%.2f" % (tmp_psnr,tmp_ssim,time_cost))
        cv2.imwrite(result_dir + file_name, img_restored)
        psnr_sum = psnr_sum + tmp_psnr
        time_sum = time_sum + time_cost
        ssim_sum = ssim_sum + tmp_ssim
        image_num = image_num + 1.0
        error_list.append(tmp_psnr)
    print("*************************************************************")
    print("平均psnr:%.2f, ssim:%.2f,time:%.2f" % (psnr_sum / image_num, ssim_sum / image_num, time_sum / image_num))
    print("*************************************************************")

    error_list.append(psnr_sum / image_num)
    error_list.append(ssim_sum / image_num)
    error_list.append(time_sum / image_num)

    return error_list
