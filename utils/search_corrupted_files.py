from torch.utils.data import DataLoader

from detection.dataloader import CaltechDatasetLoader
from settings.config import ConfigObjectDetection
from settings.dataset_network_configs import dataset_configs
from utils import collate_fn, get_valid_transform

cfg = ConfigObjectDetection().parse()
dataset_config = dataset_configs(cfg)

test_dataset = CaltechDatasetLoader(dataset_config.get("test_images"),
                                    dataset_config.get("width"),
                                    dataset_config.get("height"),
                                    dataset_config.get("classes"),
                                    get_valid_transform())

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

for idx, data in enumerate(test_loader):
    images, targets = data
    print(idx)

"""
set10_V011_0834.xml
"""