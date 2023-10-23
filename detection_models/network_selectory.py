import torch

from abc import ABC, abstractmethod

from detection_models.faster_r_cnn_model import FasterRCNN
from detection_models.ssd_model import SSDModel


class BaseNetwork(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x, y):
        pass


class FasterRCNNWrapper(BaseNetwork):
    def __init__(self, dataset_cfg):
        self.model = FasterRCNN(dataset_cfg.get("num_classes"))

    def forward(self, x, y):
        return self.model(x, y)


class SSDWrapper(BaseNetwork):
    def __init__(self, dataset_cfg):
        self.model = SSDModel(
            dataset_cfg.get("num_classes"),
            dataset_cfg.get("width"),
            dataset_cfg.get("height"),
            dataset_cfg.get("nms")
        )

    def forward(self, x, y):
        return self.model(x, y)


class NetworkFactory:
    @staticmethod
    def create_network(network_type, dataset_cfg, device=None):
        if network_type == "Faster_R_CNN":
            model = FasterRCNNWrapper(dataset_cfg).model
        elif network_type == "SSD":
            model = SSDWrapper(dataset_cfg).model
        else:
            raise ValueError(f"Wrong network type was given: {network_type}")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)

        return model
