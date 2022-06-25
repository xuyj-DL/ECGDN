# -*- coding: utf-8 -*-
import numpy as np


def __Rgb2Hsi(R, G, B):
    # 归一化到[0,1]
    R /= 255
    G /= 255
    B /= 255
    eps = 1e-8
    H, S, I = 0, 0, 0
    sumRGB = R + G + B
    Min = min(R, G, B)
    S = 1 - 3 * Min / (sumRGB + eps)
    H = np.arccos((0.5 * (R + R - G - B)) / np.sqrt((R - G) * (R - G) + (R - B) * (G - B) + eps))
    if B > G:
        H = 2 * np.pi - H
    H = H / (2 * np.pi)
    if S == 0:
        H = 0
    I = sumRGB / 3
    return np.array([H, S, I], dtype=float)


def Rgb2Hsi(img):
    HSIimg = np.zeros(img.shape, dtype=float)
    width, height = img.shape[:2]
    for w in range(width):
        for h in range(height):
            HSIimg[w, h, :] = __Rgb2Hsi(img[w, h, 0], img[w, h, 1], img[w, h, 2])
    return HSIimg


def __Hsi2Rgb(H, S, I):
    pi3 = np.pi / 3
    # 扩充弧度范围[0,2pi]
    H *= 2 * np.pi
    if H >= 0 and H < 2 * pi3:
        # [0,2pi/3)对应红->绿
        B = I * (1 - S)
        R = I * (1 + S * np.cos(H) / np.cos(pi3 - H))
        G = 3 * I - (R + B)
    elif H >= 2 * pi3 and H <= 4 * pi3:
        # [2pi/3,4pi/3)对应绿->蓝
        H = H - 2 * pi3
        R = I * (1 - S)
        G = I * (1 + S * np.cos(H) / np.cos(pi3 - H))
        B = 3 * I - (R + G)
    else:
        # [4pi/3,2pi)对应蓝->红
        H = H - 4 * pi3
        G = I * (1 - S)
        B = I * (1 + S * np.cos(H) / np.cos(pi3 - H))
        R = 3 * I - (B + G)
    return (np.array([R, G, B]) * 255).astype(np.uint8)


def Hsi2Rgb(img):
    RGBimg = np.zeros(img.shape, dtype=np.uint8)
    width, height = img.shape[:2]
    for w in range(width):
        for h in range(height):
            RGBimg[w, h, :] = __Hsi2Rgb(img[w, h, 0], img[w, h, 1], img[w, h, 2])
    return RGBimg


import matplotlib.pyplot as plt


def plot_multi_pic(rows, cols, imgs):
    for index, img in enumerate(imgs):
        plt.subplot(rows, cols, index + 1)
        plt.title(img['title'])
        plt.xticks([])  # remove ticks
        plt.yticks([])
        if img['gray']:
            plt.imshow(img['img'], cmap='gray')
        else:
            plt.imshow(img['img'])
        # plt.savefig('./{}.jpg'.format(time.time()))

    plt.show()
    # plt.savefig('testblueline.jpg')

def plot_single_pic(img,path,gray):
    #plt.title('map')
    # if gray:
    #     plt.imshow(img, cmap='gray')
    # else:
    #     plt.imshow(img)
    # fig = plt.gcf()
    #
    # plt.axis('off')
    #
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    #
    # plt.margins(0, 0)
    #
    # fig.savefig(path)
    fig, ax = plt.subplots()
    # im = img[:, :, (2, 1, 0)]
    ax.imshow(img, aspect='equal')
    plt.axis('off')
    # 去除图像周围的白边

    if gray:
        height, width = img.shape

        # plt.imshow(img, cmap='gray')
    else:
        height, width, chanels = img.shape
        # plt.imshow(img)
    # height, width,chanels = img.shape
    # 如果dpi=300，那么图像大小=height*width
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(path, dpi=300)




class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) >
                0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


import colorsys
from PIL import Image

import colorsys

def HSVColor(img):
    if isinstance(img,Image.Image):
        r,g,b = img.split()
        Hdat = []
        Sdat = []
        Vdat = []
        for rd,gn,bl in zip(r.getdata(),g.getdata(),b.getdata()) :
            h,s,v = colorsys.rgb_to_hsv(rd/255.,gn/255.,bl/255.)
            Hdat.append(int(h*255.))
            Sdat.append(int(s*255.))
            Vdat.append(int(v*255.))
        r.putdata(Hdat)
        g.putdata(Sdat)
        b.putdata(Vdat)
        return Image.merge('RGB',(r,g,b))
    else:
        return None



import time,cv2,numpy
def img2hv(img):
    img = Image.open('F:\workspace\pycharmProject\datasets\OTS_ALPHA\B\8253_0.95_0.2.jpg').convert('RGB')
    img.save('./a.jpg')
    start = time.time()
    img2 = HSVColor(img)
    print(time.time()-start)
    img2.save('./b.jpg')


    start = time.time()

    ii = cv2.imread('F:\workspace\pycharmProject\datasets\OTS_ALPHA\B\8253_0.95_0.2.jpg')
    # cv2.imshow(ii)

    iihsv = cv2.cvtColor(numpy.asarray(img),cv2.COLOR_BGR2HSV)
    print(time.time()-start)

    cv2.imshow('dd',iihsv)
    cv2.waitKey(0)
    # a = Image.open('/tmp/a.jpg')
    # b = HSVColor(a)
    # b.save('/tmp/b.jpg')
if __name__ == '__main__':
    # rename()
    img2hv('dd')