from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import random
import numpy as np
import cv2
import torch.utils.data as data
from MOC_utils.gaussian_hm import gaussian_radius, draw_umich_gaussian
from ACT_utils.ACT_aug import apply_distort, apply_expand, crop_image
## MODIFY FOR PYTORCH 1+
#cv2.setNumThreads(0)
i_file = 0


class Sampler(data.Dataset):
    def __getitem__(self, id):
        v, frame = self._indices[id]
        K = self.K
        num_classes = self.num_classes
        input_h = self._resize_height
        input_w = self._resize_width
        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        # read images
        if self._ninput > 1:
            images = [cv2.imread(self.flowfile(v, min(frame + i, self._nframes[v]))).astype(np.float32) for i in range(K + self._ninput - 1)]
        else:
            images = [cv2.imread(self.imagefile(v, frame + i)).astype(np.float32) for i in range(K)]
        data = [np.empty((3 * self._ninput, self._resize_height, self._resize_width), dtype=np.float32) for i in range(K)]

        if self.mode == 'train':
            do_mirror = random.getrandbits(1) == 1
            # filp the image
            if do_mirror:
                images = [im[:, ::-1, :] for im in images]
                if self._ninput > 1:
                    for i in range(K + self._ninput - 1):
                        images[i][:, :, 2] = 255 - images[i][:, :, 2]
            h, w = self._resolution[v]
            gt_bbox = {}
            for ilabel, tubes in self._gttubes[v].items():
                for t in tubes:
                    if frame not in t[:, 0]:
                        continue
                    assert frame + K - 1 in t[:, 0]
                    # copy otherwise it will change the gt of the dataset also
                    t = t.copy()
                    if do_mirror:
                        # filp the gt bbox
                        xmin = w - t[:, 3]
                        t[:, 3] = w - t[:, 1]
                        t[:, 1] = xmin
                    boxes = t[(t[:, 0] >= frame) * (t[:, 0] < frame + K), 1:5]

                    assert boxes.shape[0] == K
                    if ilabel not in gt_bbox:
                        gt_bbox[ilabel] = []
                    # gt_bbox[ilabel] ---> a list of numpy array, each one is K, x1, x2, y1, y2
                    gt_bbox[ilabel].append(boxes)

            # # DEBUG
            # if ilabel == 0:
            #     import os
            #     global i_file
            #     if not os.path.exists('data_aug/%d' % i_file):
            #         os.makedirs('data_aug/%d' % i_file)

            #     vis_data_bbox(images, gt_bbox, 'sample')


            # apply data augmentation
            images = apply_distort(images, self.distort_param)
            # images, gt_bbox = apply_expand(images, gt_bbox, self.expand_param, self._mean_values)
            # images, gt_bbox = crop_image(images, gt_bbox, self.batch_samplers)
            images, gt_bbox = expand_and_crop(images, gt_bbox, self._mean_values)  # my augmentation
            # # debug
            # if ilabel == 0:
            #     vis_data_bbox(images, gt_bbox, 'sample_distort_expCrop')
        else:
            # no data augmentation or flip when validation
            gt_bbox = {}
            for ilabel, tubes in self._gttubes[v].items():
                for t in tubes:
                    if frame not in t[:, 0]:
                        continue
                    assert frame + K - 1 in t[:, 0]
                    t = t.copy()
                    boxes = t[(t[:, 0] >= frame) * (t[:, 0] < frame + K), 1:5]
                    assert boxes.shape[0] == K
                    if ilabel not in gt_bbox:
                        gt_bbox[ilabel] = []
                    gt_bbox[ilabel].append(boxes)

        original_h, original_w = images[0].shape[:2]

        ## 保持比例缩放
        for i in range(len(images)):
            images[i], left, top, ratio = resize_img_keep_ratio(images[i], (input_h, input_w))
        real_w = int(original_w * ratio)
        real_h = int(original_h * ratio)


        # 将目标框与padding、resize之后提取到的特征图对应
        for ilabel in gt_bbox:
            for itube in range(len(gt_bbox[ilabel])):
                gt_bbox[ilabel][itube][:, 0] = (gt_bbox[ilabel][itube][:, 0] / original_w * real_w + left) /input_w * output_w
                gt_bbox[ilabel][itube][:, 1] = (gt_bbox[ilabel][itube][:, 1] / original_h * real_h + top) /input_h * output_h
                gt_bbox[ilabel][itube][:, 2] = (gt_bbox[ilabel][itube][:, 2] / original_w * real_w + left) /input_w * output_w
                gt_bbox[ilabel][itube][:, 3] = (gt_bbox[ilabel][itube][:, 3] / original_h * real_h + top) /input_h * output_h
        # # debug
        # if ilabel == 0:
        #     vis_data_bbox(images, gt_bbox, 'resize',resize=True)
        #     i_file += 1  
            
        # transpose image channel and normalize
        mean = np.tile(np.array(self.opt.mean, dtype=np.float32)[:, None, None], (self._ninput, 1, 1))
        std = np.tile(np.array(self.opt.std, dtype=np.float32)[:, None, None], (self._ninput, 1, 1))
        for i in range(K):
            for ii in range(self._ninput):
                data[i][3 * ii:3 * ii + 3, :, :] = np.transpose(images[i + ii], (2, 0, 1))

            data[i] = ((data[i] / 255.) - mean) / std

        # draw ground truth
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, K * 2), dtype=np.float32)
        mov = np.zeros((self.max_objs, K * 2), dtype=np.float32)
        index = np.zeros((self.max_objs), dtype=np.int64)
        index_all = np.zeros((self.max_objs, K * 2), dtype=np.int64)
        mask = np.zeros((self.max_objs), dtype=np.uint8)

        num_objs = 0
        for ilabel in gt_bbox:
            if ilabel >= num_classes:
                break
            for itube in range(len(gt_bbox[ilabel])):
                key = K // 2
                # key frame's bbox height and width （both on the feature map）
                key_h, key_w = gt_bbox[ilabel][itube][key, 3] - gt_bbox[ilabel][itube][key, 1], gt_bbox[ilabel][itube][key, 2] - gt_bbox[ilabel][itube][key, 0]
                # create gaussian heatmap
                radius = gaussian_radius((math.ceil(key_h), math.ceil(key_w)))
                radius = max(0, int(radius))

                # ground truth bbox's center in key frame
                center = np.array([(gt_bbox[ilabel][itube][key, 0] + gt_bbox[ilabel][itube][key, 2]) / 2, (gt_bbox[ilabel][itube][key, 1] + gt_bbox[ilabel][itube][key, 3]) / 2], dtype=np.float32)
                center_int = center.astype(np.int32)
                # debug
                if 0 <= center_int[0] and center_int[0] <= output_w and 0 <= center_int[1] and center_int[1] <= output_h:
                    pass
                else:
                    print('center_int:',center_int)
                    print('output_w',output_w)
                    print('output_h',output_h)
                    print('key',key)
                    # print('data',data)
                    print('v', v)
                    print('frame', frame)

                assert 0 <= center_int[0] and center_int[0] <= output_w and 0 <= center_int[1] and center_int[1] <= output_h

                # draw ground truth gaussian heatmap at each center location
                draw_umich_gaussian(hm[ilabel], center_int, radius)

                for i in range(K):
                    center_all = np.array([(gt_bbox[ilabel][itube][i, 0] + gt_bbox[ilabel][itube][i, 2]) / 2,  (gt_bbox[ilabel][itube][i, 1] + gt_bbox[ilabel][itube][i, 3]) / 2], dtype=np.float32)
                    center_all_int = center_all.astype(np.int32)
                    # wh is ground truth bbox's height and width in i_th frame
                    wh[num_objs, i * 2: i * 2 + 2] = 1. * (gt_bbox[ilabel][itube][i, 2] - gt_bbox[ilabel][itube][i, 0]), 1. * (gt_bbox[ilabel][itube][i, 3] - gt_bbox[ilabel][itube][i, 1])
                    # mov is ground truth movement from i_th frame to key frame
                    mov[num_objs, i * 2: i * 2 + 2] = (gt_bbox[ilabel][itube][i, 0] + gt_bbox[ilabel][itube][i, 2]) / 2 - \
                        center_int[0],  (gt_bbox[ilabel][itube][i, 1] + gt_bbox[ilabel][itube][i, 3]) / 2 - center_int[1]
                    # index_all are all frame's bbox center position
                    index_all[num_objs, i * 2: i * 2 + 2] = center_all_int[1] * output_w + center_all_int[0], center_all_int[1] * output_w + center_all_int[0]
                # index is key frame's boox center position
                index[num_objs] = center_int[1] * output_w + center_int[0]
                # mask indicate how many objects in this tube
                mask[num_objs] = 1
                num_objs = num_objs + 1
        result = {'input': data, 'hm': hm, 'mov': mov, 'wh': wh, 'mask': mask, 'index': index, 'index_all': index_all}

        return result

# 封装resize函数
def resize_img_keep_ratio(img,target_size):
    old_size= img.shape[0:2] # 原始图像大小
    ratio = min(float(target_size[i])/(old_size[i]) for i in range(len(old_size))) # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
    new_size = tuple([int(i*ratio) for i in old_size]) # 根据上边求得的比例计算在保持比例前提下得到的图像大小
    img = cv2.resize(img,(new_size[1], new_size[0])) # 根据上边的大小进行放缩
    pad_w = target_size[1] - new_size[1] # 计算需要填充的像素数目（图像的宽这一维度上）
    pad_h = target_size[0] - new_size[0] # 计算需要填充的像素数目（图像的高这一维度上）
    top,bottom = pad_h//2, pad_h-(pad_h//2)
    left,right = pad_w//2, pad_w -(pad_w//2)
    img_new = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,None,(0,0,0)) 
    return img_new, left, top, ratio

def adjust_bboxes(bboxes, left, top, ratio):
    """
    Adjust an array of bounding boxes based on the resizing and padding information.

    :param bboxes: a NumPy array of shape (N, 4) where each row is [x_min, y_min, x_max, y_max]
    :param left: number of pixels padded on the left side.
    :param top: number of pixels padded on the top side.
    :param ratio: the scaling ratio used during resizing.
    
    :return: a NumPy array of shape (N, 4) with adjusted bounding boxes.
    """
    adjusted_bboxes = np.zeros_like(bboxes)

    for i in range(bboxes.shape[0]):
        x_min, y_min, x_max, y_max = bboxes[i]
        
        # Scale the bounding box coordinates
        new_x_min = int(x_min * ratio)
        new_y_min = int(y_min * ratio)
        new_x_max = int(x_max * ratio)
        new_y_max = int(y_max * ratio)
        
        # Add the padding offsets
        new_x_min += left
        new_y_min += top
        new_x_max += left
        new_y_max += top
        
        adjusted_bboxes[i] = [new_x_min, new_y_min, new_x_max, new_y_max]

    return adjusted_bboxes

def expand_and_crop(images, gt_bbox, _mean_values):
    """
    将图像宽高随机扩展0~0.1，然后裁剪原始图像大小
    images: list[7] -> array(H, W, 3) float32
    gt_bbox: dict[1] -> list[1] -> array(7, 4) float32
    """
    if random.random() < 0.5:
        w_h_expand_ratio = 0.1

        ori_h, ori_w, _ = images[0].shape
        pad_h = int(w_h_expand_ratio * ori_h)
        pad_w = int(w_h_expand_ratio * ori_w)

        # 上下左右扩展
        for i in range(len(images)):
            new_img = np.zeros((pad_h * 2 + ori_h, pad_w * 2 + ori_w, 3), dtype=np.float32) + np.array(_mean_values).reshape(1, 1, 3)
            new_img[pad_h: pad_h + ori_h, pad_w: pad_w + ori_w, :] = images[i]
            images[i] = new_img

        for cls in gt_bbox.keys():
            for i in range(len(gt_bbox[cls])):
                gt_bbox[cls][i] = gt_bbox[cls][i] + np.array([[pad_w, pad_h, pad_w, pad_h]])  # x1, y1, x2, y2

        # 随机裁剪(ori_h, ori_w)大小的图像
        crop_x = np.random.randint(2 * pad_w)
        crop_y = np.random.randint(2 * pad_h)
        for i in range(len(images)):
            images[i] = images[i][crop_y: crop_y + ori_h, crop_x: crop_x + ori_w, :]

        for cls in gt_bbox.keys():
            for i in range(len(gt_bbox[cls])):
                gt_bbox[cls][i] = gt_bbox[cls][i] - np.array([[crop_x, crop_y, crop_x, crop_y]])
                gt_bbox[cls][i] = np.maximum(gt_bbox[cls][i], 0)
                gt_bbox[cls][i][:, 0::2] = np.minimum(gt_bbox[cls][i][:, 0::2], ori_w - 1)
                gt_bbox[cls][i][:, 1::2] = np.minimum(gt_bbox[cls][i][:, 1::2], ori_h - 1)

    return images, gt_bbox


def vis_data_bbox(images, gt_bbox, img_name, resize=False):
    for i in range(len(images)):
        img = images[i].astype(np.uint8)  # (H, W, 3)
        if resize:
            img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))

        k = list(gt_bbox.keys())[0]

        bbox = gt_bbox[k][0][i].astype(int)  # [4,]
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), thickness=2)
        cv2.imwrite('data_aug/%d/%s_%d.png' % (i_file, img_name, i), img)