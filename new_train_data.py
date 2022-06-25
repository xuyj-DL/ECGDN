"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: train_data.py
about: build the training dataset
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import os
import random
from utils1.canny import image_to_edge
import torch
import cv2
import numpy as np


# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir, sgima, is_color=False):
        super().__init__()

        self.haze_imgs_dir = os.path.join(train_data_dir, 'B')
        self.gt_imgs_dir = os.path.join(train_data_dir, 'clear')

        haze_names = []

        for file_name in os.listdir(self.haze_imgs_dir):
            haze_names.append(file_name)

        self.haze_names = haze_names
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir
        self.sigma = sgima

        self.is_color = is_color

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        haze_name = self.haze_names[index]

        haze_img = Image.open(os.path.join(self.haze_imgs_dir, haze_name))

        gt_name = haze_name.split('_')[0]

        # gt_name = haze_name.split('.')[0]
        # print(gt_name)
        try:
            gt_img = Image.open(os.path.join(self.gt_imgs_dir, gt_name + '.jpg'))
        except:
            gt_img = Image.open(os.path.join(self.gt_imgs_dir, gt_name + '.png')).convert('RGB')

        width, height = haze_img.size

        if width < crop_width or height < crop_height:
            raise Exception('Bad image size: {}'.format(gt_name))

        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_crop_img)
        gt = transform_gt(gt_crop_img)

        # --- Check the channel is 3 or not --- #
        if list(haze.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
            raise Exception('Bad image channel: {}{}'.format(haze_name, gt_name))

        haze_edge, haze_gray = image_to_edge(haze, sigma=self.sigma)
        gt_edge, gt_gray = image_to_edge(gt, sigma=self.sigma)

        haze_edge = torch.cat((haze_edge, haze_gray), dim=0)

        if self.is_color:
            # cv2.imshow(haze_crop_img)
            # cv2.waitKey(0)
            haze_hsv = cv2.cvtColor(np.asarray(haze_crop_img), cv2.COLOR_RGB2HSV)
            gt_hsv = cv2.cvtColor(np.asarray(gt_crop_img), cv2.COLOR_RGB2HSV)

            # cv2.imwrite('a.jpg', haze_hsv)
            # # cv2.imwrite('a.jpg', haze_hsv)
            # haze_crop_img.save('b.jpg')
            # print(type(haze_hsv), type(haze_crop_img))
            transform_haze_hsv = Compose([ToTensor()])
            transform_gt_hsv = Compose([ToTensor()])

            return haze, gt, haze_edge, gt_edge, transform_haze_hsv(haze_hsv[:, :, 0:2]), transform_gt_hsv(
                gt_hsv[:, :, 0:2])

        return haze, gt, haze_edge, gt_edge

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)


# --- Training dataset --- #


class TrainDataSimple(data.Dataset):
    def __init__(self, crop_size, train_data_dir, sgima):
        super().__init__()

        self.haze_imgs_dir = os.path.join(train_data_dir, 'B')
        self.gt_imgs_dir = os.path.join(train_data_dir, 'clear')

        haze_names = []

        for file_name in os.listdir(self.haze_imgs_dir):
            haze_names.append(file_name)

        self.haze_names = haze_names
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir
        self.sigma = sgima

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        haze_name = self.haze_names[index]

        haze_img = Image.open(os.path.join(self.haze_imgs_dir, haze_name))

        gt_name = haze_name.split('_')[0]
        # print(gt_name)
        try:
            gt_img = Image.open(os.path.join(self.gt_imgs_dir, gt_name + '.jpg'))
        except:
            gt_img = Image.open(os.path.join(self.gt_imgs_dir, gt_name + '.png')).convert('RGB')

        width, height = haze_img.size

        if width < crop_width or height < crop_height:
            raise Exception('Bad image size: {}'.format(gt_name))

        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_crop_img)
        gt = transform_gt(gt_crop_img)

        # --- Check the channel is 3 or not --- #
        if list(haze.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
            raise Exception('Bad image channel: {}{}'.format(haze_name, gt_name))


        return haze, gt

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)


class ValData(data.Dataset):
    def __init__(self, val_data_dir, sgima,is_single=False, tail_format='', slice='_', is_color=False, crop_size=None):
        super().__init__()

        self.haze_imgs_dir = os.path.join(val_data_dir, 'B')
        self.gt_imgs_dir = os.path.join(val_data_dir, 'clear')

        haze_names = []

        for file_name in os.listdir(self.haze_imgs_dir):
            haze_names.append(file_name)

        self.haze_names = haze_names
        self.val_data_dir = val_data_dir
        self.sigma = sgima
        self.tail_format = tail_format
        self.slice = slice

        self.is_color = is_color
        self.crop_size = crop_size
        self.is_single = is_single

    def get_images(self, index):
        haze_name = self.haze_names[index]

        haze_img = Image.open(os.path.join(self.haze_imgs_dir, haze_name))
        if self.is_single:
            gt_img = Image.open(os.path.join(self.haze_imgs_dir, haze_name))
        else:
            gt_name = haze_name.split(self.slice)[0]
            gt_name = os.path.join(self.gt_imgs_dir, gt_name + self.tail_format)
            try:
                gt_img = Image.open(gt_name + '.jpg')
            except:
                gt_img = Image.open(gt_name + '.png').convert('RGB')


        if self.crop_size:
            width, height = haze_img.size
            crop_width, crop_height = self.crop_size
            x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
            haze_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
            gt_img = gt_img.crop((x, y, x + crop_width, y + crop_height))


        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_img)
        gt = transform_gt(gt_img)

        # --- Check the channel is 3 or not --- #
        if list(haze.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
            raise Exception('Bad image channel: {}{}'.format(haze_name, gt_name))

        haze_edge, haze_gray = image_to_edge(haze, sigma=self.sigma)

        haze_edge = torch.cat((haze_edge, haze_gray), dim=0)


        if self.is_color:
            # cv2.imshow(haze_crop_img)
            # cv2.waitKey(0)
            haze_hsv = cv2.cvtColor(np.asarray(haze_img), cv2.COLOR_RGB2HSV)

            # cv2.imwrite('a.jpg', haze_hsv)
            # # cv2.imwrite('a.jpg', haze_hsv)
            # haze_crop_img.save('b.jpg')
            # print(type(haze_hsv), type(haze_crop_img))
            transform_haze_hsv = Compose([ToTensor()])

            return haze, gt, haze_edge, haze_name, transform_haze_hsv(haze_hsv[:, :, 0:2])

        return haze, gt, haze_edge, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)
