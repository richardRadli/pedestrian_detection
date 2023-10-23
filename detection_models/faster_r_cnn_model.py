import torchvision
import torch.nn as nn

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNN(nn.Module):
    def __init__(self, num_classes) -> None:
        super(FasterRCNN, self).__init__()
        self.num_classes = num_classes
        self.model = self.build_model()

    def forward(self, x, y):
        return self.model(x, y)

    def build_model(self):
        # load Faster RCNN pre-trained model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

        # get the number of input features
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # define a new head for the detector with required number of classes
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        return model
