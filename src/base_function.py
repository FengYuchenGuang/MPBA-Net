import logging,torch,os,cv2,skimage
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from medpy.metric.binary import hd, dc, assd, jc
from src.losses import dice_loss
import numpy as np
from PIL import Image


def get_logger(name,txt_path):
    logger = logging.getLogger(name)

    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s')

    file_handler = logging.FileHandler(txt_path)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    # logger.addHandler(stream_handler)
    return logger


def ce_loss(pred, gt):
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    return (-gt * torch.log(pred) - (1 - gt) * torch.log(1 - pred)).mean()



def structure_loss(pred, mask):
    """            TransFuse train loss        """
    """            Without sigmoid             """
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.65,  #0.6
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    p = inputs
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="mean")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def train(epoch,
          model,
          train_loader,
          parse_config,
          hybrid_loss,
          optimizer,
          writer,
          logging_loss):
    model.train()
    iteration = 0
    for batch_idx, batch_data in enumerate(train_loader):
        #         print(epoch, batch_idx)
        data = batch_data['image'].cuda().float()
        label = batch_data['label'].cuda().float()
        point = (batch_data['point'] > 0).cuda().float()
        #point_All = (batch_data['point_All'] > 0).cuda().float()

        if parse_config.net_layer == 18:
            point_c4 = nn.functional.max_pool2d(point,
                                                kernel_size=(16, 16),
                                                stride=(16, 16))
            point = nn.functional.max_pool2d(point,
                                             kernel_size=(8, 8),
                                             stride=(8, 8))
        else:
            point_c5 = nn.functional.max_pool2d(point,
                                                kernel_size=(32, 32),
                                                stride=(32, 32))

            point_c4 = nn.functional.max_pool2d(point,
                                                kernel_size=(16, 16),
                                                stride=(16, 16))

        if parse_config.point_pred == 1:
            model.eval()
            output, point_maps_pre = model(data)
            output = torch.sigmoid(output)

            #print("point_pre shape:{}, point shape:{}".format(point_pre.shape,point.shape))
            assert (output.shape == label.shape)
            loss_dc = dice_loss(output, label)
            #print(point_maps_pre[-1].shape, point_c4.shape)
            assert (point_maps_pre[-1].shape == point_c4.shape)

            point_loss = 0.
            for i in range(len(point_maps_pre)):
                point_loss += hybrid_loss(point_maps_pre[i], point_c4)
            point_loss = point_loss / len(point_maps_pre)

            loss = loss_dc + point_loss  # point_loss weight: 3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration = iteration + 1

            if (batch_idx + 1) % 10 == 0:
                writer.add_scalar('loss/dc_loss', loss_dc,
                                  batch_idx + epoch * len(train_loader))
                writer.add_scalar('loss/point_loss', point_loss,
                                  batch_idx + epoch * len(train_loader))
                writer.add_scalar('loss/loss', loss,
                                  batch_idx + epoch * len(train_loader))

                writer.add_image('label', label[0],
                                 batch_idx + epoch * len(train_loader))
                writer.add_image('output', output[0] > 0.5,
                                 batch_idx + epoch * len(train_loader))
                writer.add_image('point', point_c4[0],
                                 batch_idx + epoch * len(train_loader))
                writer.add_image('point_pre', point_maps_pre[-1][0],
                                 batch_idx + epoch * len(train_loader))

                logging_loss.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
    print("Iteration numbers: ", iteration)



def evaluation(epoch,
               loader,
               model,
               parse_config,
               hybrid_loss,
               writer,
               logging_evaluation):
    model.eval()
    dice_value = 0
    iou_value = 0
    dice_average = 0
    iou_average = 0
    numm = 0
    for batch_idx, batch_data in enumerate(loader):
        data = batch_data['image'].cuda().float()
        label = batch_data['label'].cuda().float()
        point = (batch_data['point'] > 0).cuda().float()
        point_c5 = nn.functional.max_pool2d(point,
                                            kernel_size=(32, 32),
                                            stride=(32, 32))
        point_c4 = nn.functional.max_pool2d(point,
                                            kernel_size=(16, 16),
                                            stride=(16, 16))

        with torch.no_grad():
            if parse_config.arch == 'transfuse':
                _, _, output = model(data)
                loss_fuse = structure_loss(output, label)
            elif parse_config.point_pred == 0:
                output = model(data)
            elif parse_config.cross == 1 and parse_config.point_pred == 1:
                output, point_maps_pre_1, point_maps_pre_2 = model(data)
                point_loss_c4 = 0.
                for i in range(len(point_maps_pre_1) - 1):
                    point_loss_c4 += hybrid_loss(point_maps_pre_1[i], point_c4)
                point_loss_c4 = 1.0 / len(point_maps_pre_1) * (
                    point_loss_c4 + hybrid_loss(point_maps_pre_1[-1], point_c4))
                point_loss_c5 = 0.
                for i in range(len(point_maps_pre_2) - 1):
                    point_loss_c5 += hybrid_loss(point_maps_pre_2[i], point_c5)
                point_loss_c5 = 1.0 / len(point_maps_pre_2) * (
                    point_loss_c5 + hybrid_loss(point_maps_pre_2[-1], point_c5))
                point_loss = 0.5 * (point_loss_c4 + point_loss_c5)
            elif parse_config.point_pred == 1:
                output, point_maps_pre = model(data)
                point_loss = 0.
                for i in range(len(point_maps_pre) - 1):
                    point_loss += hybrid_loss(point_maps_pre[i], point_c4)
                point_loss = 1.0 / len(point_maps_pre) * (
                    point_loss + hybrid_loss(point_maps_pre[-1], point_c4))

            output = torch.sigmoid(output)

            loss_dc = dice_loss(output, label)

            if parse_config.arch == 'transfuse':
                loss = loss_fuse
            elif parse_config.arch == 'transunet':
                loss = 0.5 * loss_dc + 0.5 * ce_loss(output, label)
            elif parse_config.point_pred == 0:
                loss = loss_dc
            elif parse_config.cross == 1 and parse_config.point_pred == 1:
                loss = loss_dc + point_loss
            elif parse_config.point_pred == 1:
                loss = loss_dc + 3 * point_loss

            output = output.cpu().numpy() > 0.5

        label = label.cpu().numpy()
        assert (output.shape == label.shape)
        dice_ave = dc(output, label)
        iou_ave = jc(output, label)
        dice_value += dice_ave
        iou_value += iou_ave
        numm += 1

    dice_average = dice_value / numm
    iou_average = iou_value / numm
    writer.add_scalar('val_metrics/val_dice', dice_average, epoch)
    writer.add_scalar('val_metrics/val_iou', iou_average, epoch)
    logging_evaluation.info("Average dice value of evaluation dataset = {:.4f}".format(dice_average))
    logging_evaluation.info("Average iou  value of evaluation dataset = {:.4f}".format(iou_average))
    print("Average dice value of evaluation dataset = ", dice_average)
    print("Average iou value of evaluation dataset = ", iou_average)
    return dice_average, iou_average, loss




