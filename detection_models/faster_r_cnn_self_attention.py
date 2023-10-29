import torch
import torch.nn as nn
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class SelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_weights = torch.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, values)
        return attended_values


class FasterRCNNSelfAttention(nn.Module):
    def __init__(self, num_classes, attention_dim=1024):
        super(FasterRCNNSelfAttention, self).__init__()
        self.num_classes = num_classes
        self.attention_dim = attention_dim

        # Load the Faster RCNN pre-trained model
        self.model = self.build_model()

    def build_model(self):
        # Load Faster RCNN pre-trained model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

        # Get the number of input features
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # Define a new head for the detector with the required number of classes
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        # Extend the TwoMLPHead with SelfAttention
        model.roi_heads.box_head.fc6 = nn.Sequential(
            model.roi_heads.box_head.fc6,
            SelfAttention(in_features, self.attention_dim)  # Add SelfAttention layer
        )

        return model

    def forward(self, x, y):
        # Pass the input through the Faster RCNN model
        x = self.model(x, y)

        return x
