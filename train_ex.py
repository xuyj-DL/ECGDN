

# --- Imports --- #
import time
import torch, warnings
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from torchvision.models import vgg16
from network1 import LossNetwork, AggregationNet
from utils import LambdaLR

import sys
import os
import pytorch_ssim
from new_train_data import TrainData, ValData, TrainDataSimple

import torchvision.utils as utils
from Unet_cat2 import UNetDouble
from new_test import to_psnr, to_ssim_skimage
import math
from torch import optim

warnings.filterwarnings('ignore')


def lr_schedule_cosdecay(t, T, init_lr):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr


def colorLoss(hs, gt_hs):
    d = gt_hs - hs
    loss = d[:, 0, :, :].pow(2) + d[:, 1, :, :].pow(2)
    loss = loss.sqrt().mean()
    return loss


def save_image(dehaze, image_name, category):
    dehaze_images = torch.split(dehaze, 1, dim=0)
    batch_num = len(dehaze_images)

    if not os.path.exists('./{}_results/'.format(category)):
        os.makedirs('./{}_results/'.format(category))
    for ind in range(batch_num):
        utils.save_image(dehaze_images[ind], './{}_results/{}'.format(category, image_name[0]))


def validation(net, val_data_loader, device, save_tag=False):
    """
    :param net: GateDehazeNet
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param category: indoor or outdoor test dataset
    :param save_tag: tag of saving image or not
    :return: average PSNR value
    """
    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():
            haze, gt, haze_edge, image_name, haze_hs = val_data

            haze = haze.to(device)
            haze_edge = haze_edge.to(device)

            gt = gt.to(device)
            haze_hs = haze_hs.to(device)

            edge, dehaze, hs = net(haze_edge, haze, haze_hs)

            # else:
            psnr_list.extend(to_psnr(dehaze, gt))

            # --- Calculate the average SSIM --- #
            ssim_list.extend(to_ssim_skimage(dehaze, gt))

            # --- Save image --- #
            if save_tag:
                save_image(dehaze, image_name, 'ITS')

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)

    return avr_psnr, avr_ssim


if __name__ == '__main__':

    # from perceptual import LossNetwork
    plt.switch_backend('agg')

    # --- Parse hyper-parameters  --- #
    parser = argparse.ArgumentParser(description='Hyper-parameters for GridDehazeNet')
    parser.add_argument('-learning_rate', help='Set the learning rate', default=4e-4, type=float)
    parser.add_argument('-crop_size', help='Set the crop_size', default=[240, 240], nargs='+', type=int)
    parser.add_argument('-train_batch_size', help='Set the training batch size', default=16, type=int)
    parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=4, type=int)
    parser.add_argument('--sigma', type=float, default=2.)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--image_root', type=str,
                        default='../../datasets/reside_ITS/train')
    # parser.add_argument('--image_root', type=str,
    #                     default='../datasets/OTS_ALPHA')
    args = parser.parse_args()

    learning_rate = args.learning_rate
    crop_size = args.crop_size
    train_batch_size = args.train_batch_size

    val_batch_size = args.val_batch_size
    val_data_dir = '../../datasets/reside_SOTS/outdoor/test'
    print('--- Hyper-parameters for training ---')

    start_epoch = 26
    num_epochs = 80
    # --- Gpu device --- #
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Define the network --- #


    from Unet_cross import UNetCrossLAColor12

    net = UNetCrossLAColor12()

    # --- Build optimizer --- #
    optimizer = torch.optim.Adam([{'params': net.parameters(), 'initial_lr': learning_rate}], lr=learning_rate, betas=(0.9, 0.999))
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                       lr_lambda=LambdaLR(num_epochs, 0, decay_start_epoch=25).step,last_epoch=25)
    # --- Multi-GPU --- #
    net = net.to(device)
    net = nn.DataParallel(net, device_ids=device_ids)

    # --- Define the perceptual loss network --- #
    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.to(device)
    for param in vgg_model.parameters():
        param.requires_grad = False

    loss_network = LossNetwork(vgg_model)
    loss_network.eval()
    # --- SSIM loss --- #
    #hf_ssim_loss = pytorch_ssim.SSIM()
    dehaze_ssim_loss = pytorch_ssim.SSIM()

    # --- Load the network weight --- #
    try:
        net.load_state_dict(torch.load('./checkpoints/epoch_{}.pk'.format(125)))
        print('--- weight loaded ---')
    except:
        print('--- no weight loaded ---')

    # --- Calculate all trainable parameters in network --- #
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    # --- Load training data and validation/test data --- #
    train_data_loader = DataLoader(TrainData(args.crop_size, args.image_root, args.sigma, is_color=True),
                                   batch_size=args.train_batch_size,
                                   shuffle=True, num_workers=min(16, train_batch_size))
    val_data_loader = DataLoader(ValData(val_data_dir, args.sigma, is_color=True), batch_size=val_batch_size,
                                 shuffle=False,
                                 num_workers=1)

    # dataset
    # train_data_loader = create_image_dataset(args)
    # print(train_data_loader.__len__())

    # --- Previous PSNR and SSIM in testing --- #
    # old_val_psnr, old_val_ssim = validation(net, val_data_loader, device, category)
    # print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))
    print(net)
    for epoch in range(start_epoch, num_epochs):
        net.train()

        for batch_id, train_data in enumerate(train_data_loader):
            
            # haze, gt = train_data
            haze, gt, haze_edge, gt_edge, haze_hs, gt_hs = train_data

            haze = haze.to(device)
            gt = gt.to(device)
            haze_edge = haze_edge.to(device)
            gt_edge = gt_edge.to(device)

            haze_hs = haze_hs.to(device)
            gt_hs = gt_hs.to(device)


            hs_input = torch.cat((haze_hs, haze), dim=1)
            #
            # # --- Zero the parameter gradients --- #
            optimizer.zero_grad()
            #
            # # --- Forward + Backward + Optimize --- #
            edge, dehaze, hs = net(haze_edge, haze, hs_input)
            #
            smooth_loss = nn.L1Loss()(dehaze, gt)
            #
            perceptual_loss = loss_network(dehaze, gt)

            edge_loss = nn.L1Loss()(edge, gt_edge)

            ssim_l = 1-dehaze_ssim_loss(dehaze,gt)
            # color loss
            color_loss = 0.5 * (colorLoss(hs, gt_hs) * 0.5 + nn.L1Loss()(hs, gt_hs))
        
            #
            loss = 1.0 * smooth_loss + 0.04 * perceptual_loss + 0.5 * edge_loss + 0.1 * color_loss + 0.05*ssim_l
            #
            loss.backward()
            optimizer.step()
            #
            # # --- To calculate average PSNR --- #
            sys.stdout.write(
                '\rEpoch %s/%s;Iteration %s/%s; loss_G: %0.6f;feature_loss:%f;perceptual_loss:%f; edge:%f;color_loss:%f;ssim:%f lr :%f ' %
                (epoch, num_epochs, batch_id, len(train_data_loader), loss.item(), smooth_loss, perceptual_loss,
                 edge_loss, color_loss,ssim_l,
                 optimizer.param_groups[0]['lr']))
            #break
        # --- Save the network parameters --- #
        torch.save(net.state_dict(), ('./checkpoints/epoch_%d.pk' % (epoch)))
       # break
        lr_scheduler_G.step()

       
 
    if False:

        print('==== testing ==== ')
        # --- Use the evaluation model in testing --- #
        net.eval()

        start_epoch = 0
        end_epoch = 80
        with open("./checkpoints/epochs_psnr_ssim.txt", "a") as f:
            for i in range(start_epoch, end_epoch):

                file_pk = './checkpoints/epoch_{}.pk'.format(i)
                if os.path.exists(file_pk):
                    # file_pk = os.path.join(output_path, i)
                    f.write('==={}===\n'.format(file_pk))

                    # --- Load the network weight --- #
                    net.load_state_dict(torch.load(file_pk))
                    # --- Use the evaluation model in testing --- #
                    net.eval()
                    print('--- Testing starts! ---', file_pk)
                    start_time = time.time()
                    val_psnr, val_ssim = validation(net, val_data_loader, device, save_tag=False)
                    print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))
                    f.write('val_psnr: {0:.2f}, val_ssim: {1:.4f}\n'.format(val_psnr, val_ssim))
