
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.resnet import ResNet18_OS16, ResNet50_OS16
from src.ASPP import ASPP, ASPP_Bottleneck


class DeepLabV3(nn.Module):
    def __init__(self, num_classes, num_layers):
        super(DeepLabV3, self).__init__()

        self.num_classes = num_classes
        layers = num_layers

        if layers == 18:
            self.resnet = ResNet18_OS16()
            self.aspp = ASPP(num_classes=self.num_classes)
        elif layers == 50:
            self.resnet = ResNet50_OS16()
            self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))
        h = x.size()[2]
        w = x.size()[3]
        feature_map = self.resnet(x)

        output = self.aspp(
            feature_map)  # (shape: (batch_size, num_classes, h/16, w/16))
        output = F.upsample(
            output, size=(h, w),
            mode="bilinear")  # (shape: (batch_size, num_classes, h, w))
        return output


if __name__ == '__main__':
    model = DeepLabV3(12).cuda()
    d = torch.rand((2, 3, 384, 384)).cuda()
    o = model(d)
    print(o.size())