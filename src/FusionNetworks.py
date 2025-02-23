import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.base import DeepLabV3 as base
from src.BAAG_BCA import BoundaryCrossAttention
from src.transformer import BoundaryAwareTransformer, Transformer
from src.ASPP import ASPP_Bottleneck

root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(root_path))
sys.path.insert(0, os.path.join(root_path, 'lib'))
sys.path.insert(0, os.path.join(root_path, 'lib/Cell_DETR_master'))

class MPBA_Net(nn.Module):
    def __init__(
            self,
            num_classes,
            num_layers,
            point_pred,
            decoder=False,
            transformer_type_index=0,
            hidden_features=128,  # 256
            number_of_query_positions=1,
            segmentation_attention_heads=8):

        super(MPBA_Net, self).__init__()

        self.num_classes = num_classes
        self.point_pred = point_pred
        self.transformer_type = "MPBA_Net" if transformer_type_index == 0 else "Transformer"
        self.use_decoder = decoder  # parse_config.ppl = 6
        self.deeplab = base(num_classes, num_layers)

        in_channels = 2048 if num_layers == 50 else 512

        self.convolution_mapping = nn.Conv2d(in_channels=in_channels,
                                             out_channels=hidden_features,
                                             kernel_size=(1, 1),
                                             stride=(1, 1),
                                             padding=(0, 0),
                                             bias=True)

        self.query_positions = nn.Parameter(data=torch.randn(
            number_of_query_positions, hidden_features, dtype=torch.float),
                                            requires_grad=True)

        self.row_embedding = nn.Parameter(data=torch.randn(100,
                                                           hidden_features //
                                                           2,
                                                           dtype=torch.float),
                                          requires_grad=True)
        self.column_embedding = nn.Parameter(data=torch.randn(
            100, hidden_features // 2, dtype=torch.float),
                                             requires_grad=True)

        self.transformer = [
            Transformer(d_model=hidden_features),
            BoundaryAwareTransformer(d_model=hidden_features)
        ][point_pred]

        if self.use_decoder:
            self.BCA = BoundaryCrossAttention(hidden_features, 8)

        self.trans_out_conv = nn.Conv2d(in_channels=hidden_features,
                                        out_channels=in_channels,
                                        kernel_size=(1, 1),
                                        stride=(1, 1),
                                        padding=(0, 0),
                                        bias=True)

        self.head = StripPooling(2048, up_kwargs={'mode': 'bilinear', 'align_corners': True})
        self.MPF = MutilPooling(2048,self.num_classes,up_kwargs={'mode': 'bilinear', 'align_corners': True})

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]
        feature_map = self.deeplab.resnet(x)
        features = self.convolution_mapping(feature_map)

        height, width = features.shape[2:]
        batch_size = features.shape[0]
        positional_embeddings = torch.cat([
            self.column_embedding[:height].unsqueeze(dim=0).repeat(
                height, 1, 1),
            self.row_embedding[:width].unsqueeze(dim=1).repeat(1, width, 1)
        ],
                                          dim=-1).permute(
                                              2, 0, 1).unsqueeze(0).repeat(
                                                  batch_size, 1, 1, 1)

        if self.transformer_type == 'MPBA_Net':
            latent_tensor, features_encoded, point_maps = self.transformer(
                features, None, self.query_positions, positional_embeddings)
        else:
            latent_tensor, features_encoded = self.transformer(
                features, None, self.query_positions, positional_embeddings)
            point_maps = []

        latent_tensor = latent_tensor.permute(2, 0, 1)

        if self.use_decoder:
            features_encoded, point_dec = self.BCA(features_encoded,latent_tensor)
            point_maps.append(point_dec)

        trans_feature_maps = self.trans_out_conv(features_encoded.contiguous())  #.contiguous()
        trans_feature_maps = trans_feature_maps + feature_map

        output = self.MPF(trans_feature_maps) # (shape: (batch_size, num_classes, h/16, w/16))
        output = F.interpolate(output, size=(h, w),mode="bilinear")  # (shape: (batch_size, num_classes, h, w))


        if self.point_pred == 1:
            return output, point_maps

        return output


class StripPooling(nn.Module):
    def __init__(self, in_channels, up_kwargs={'mode': 'bilinear', 'align_corners': True}):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))#1*W
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))#H*1
        inter_channels = int(in_channels / 4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                     nn.BatchNorm2d(inter_channels))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                     nn.BatchNorm2d(inter_channels))
        self.conv4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU(True))
        self.conv5 = nn.Sequential(nn.Conv2d(inter_channels, in_channels, 1, bias=False),
                                   nn.BatchNorm2d(in_channels))
        self._up_kwargs = up_kwargs

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]
        x1 = self.conv1(x)
        x2 = F.interpolate(self.conv2(self.pool1(x1)), (h, w), **self._up_kwargs)  # 1*W part of the structure picture
        x3 = F.interpolate(self.conv3(self.pool2(x1)), (h, w), **self._up_kwargs)  # H*1 part of the structure picture
        x4 = self.conv4(F.relu_(x2 + x3))
        out = self.conv5(x4)
        return F.relu_(x + out)

class MutilPooling(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 up_kwargs={'mode': 'bilinear', 'align_corners': True}
                 ):
        super(MutilPooling, self).__init__()
        self.num_classes = num_classes
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))#1*W
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))#H*1
        inter_channels = int(in_channels / 4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                     nn.BatchNorm2d(inter_channels))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                     nn.BatchNorm2d(inter_channels))
        self.conv4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU(True))
        self.conv5 = nn.Sequential(nn.Conv2d(inter_channels, in_channels, 1, bias=False),
                                   nn.BatchNorm2d(in_channels))

        self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)

        self.conv_1x1_1 = nn.Conv2d(4*512, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(4*512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(4*512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(4*512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(4*512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

        self.conv_1x1_4 = nn.Conv2d(256, num_classes, kernel_size=1)

        self._up_kwargs = up_kwargs

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]
        x1 = self.conv1(x)
        x2 = F.interpolate(self.conv2(self.pool1(x1)), (h, w), **self._up_kwargs)
        x3 = F.interpolate(self.conv3(self.pool2(x1)), (h, w), **self._up_kwargs)

        x4 = self.conv4(F.relu_(x2 + x3))
        out = self.conv5(x4)
        feature_map = F.relu_(x + out)

        feature_map_h = feature_map.size()[2] # (== h/16)
        feature_map_w = feature_map.size()[3] # (== w/16)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # (shape: (batch_size, 256, h/16, w/16))

        out_img = self.avg_pool(feature_map) # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/16, w/16))

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 1280, h/16, w/16))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/16, w/16))
        out = self.conv_1x1_4(out) # (shape: (batch_size, num_classes, h/16, w/16))

        return out



