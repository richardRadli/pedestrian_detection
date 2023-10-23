import torchvision
import torch.nn as nn
from torchvision.models.detection.ssd import SSD, DefaultBoxGenerator, SSDHead


class SSDModel(nn.Module):
    def __init__(self, num_classes, size_w, size_h, nms: float = 0.45):
        super(SSDModel, self).__init__()
        self.num_classes = num_classes
        self.size_w = size_w
        self.size_h = size_h
        self.nms = nms
        self.model = self.create_model()

    def forward(self, x, y):
        return self.model(x, y)

    def create_model(self):
        model_backbone = torchvision.models.resnet34(
            weights=torchvision.models.ResNet34_Weights.DEFAULT
        )
        conv1 = model_backbone.conv1
        bn1 = model_backbone.bn1
        relu = model_backbone.relu
        max_pool = model_backbone.maxpool
        layer1 = model_backbone.layer1
        layer2 = model_backbone.layer2
        layer3 = model_backbone.layer3
        layer4 = model_backbone.layer4
        backbone = nn.Sequential(
            conv1, bn1, relu, max_pool,
            layer1, layer2, layer3, layer4
        )
        out_channels = [512, 512, 512, 512, 512, 512]
        anchor_generator = DefaultBoxGenerator(
            [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        )
        num_anchors = anchor_generator.num_anchors_per_location()
        head = SSDHead(out_channels, num_anchors, self.num_classes)
        model = SSD(
            backbone=backbone,
            num_classes=self.num_classes,
            anchor_generator=anchor_generator,
            size=(self.size_w, self.size_h),
            head=head,
            nms_thresh=self.nms
        )
        return model
