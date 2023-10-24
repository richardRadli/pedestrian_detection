import colorama
import logging
import torch
import typing

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.classification import Precision, Recall, F1Score
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from dataloader import CaltechDatasetLoader
from detection_models.network_selectory import NetworkFactory
from settings.config import ConfigObjectDetection
from settings.dataset_network_configs import dataset_configs, network_configs
from utils.utils import collate_fn, get_valid_transform, find_latest_file_in_latest_directory


class EvalObjectDetectionModel:
    def __init__(self):
        colorama.init()

        self.cfg = ConfigObjectDetection().parse()

        self.dataset_config = dataset_configs(self.cfg)
        network_config = network_configs(self.cfg)

        latest_model_file = find_latest_file_in_latest_directory(network_config.get("weights_folder"))

        test_dataset = CaltechDatasetLoader(self.dataset_config.get("test_images"),
                                            self.dataset_config.get("width"),
                                            self.dataset_config.get("height"),
                                            self.dataset_config.get("classes"),
                                            get_valid_transform())

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = NetworkFactory().create_network(self.cfg.type_of_net, self.dataset_config, device=self.device)

        checkpoint = torch.load(latest_model_file, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)

    @staticmethod
    def mean_avg_precision(predictions: typing.List, gt: typing.List):
        metric = MeanAveragePrecision()
        metric.update(predictions, gt)
        mAP = metric.compute()

        logging.info(mAP)

        return mAP

    def precision(self, predictions, gt):
        p = Precision(task="multiclass", average='macro', num_classes=self.dataset_config.get("num_classes"))
        res = p(predictions, gt)
        logging.info(f"P: {res}")

        return res

    def recall(self, predictions, gt):
        recall = Recall(task="multiclass", average='macro', num_classes=self.dataset_config.get("num_classes"))
        res = recall(predictions, gt)
        logging.info(f"R: {res}")

        return res

    def f1_score(self, predictions, gt):
        f1 = F1Score(task="multiclass", num_classes=self.dataset_config.get("num_classes"))
        res = f1(predictions, gt)
        logging.info(f"F1 score: {res}")

        return res

    def validate(self):
        self.model.eval()

        # Initialize tqdm progress bar.
        prog_bar = tqdm(self.test_loader, total=len(self.test_loader))
        gt = []
        preds = []
        preds2 = []

        for _, data in enumerate(prog_bar):
            images, targets = data
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                outputs = self.model(images, targets)

            # For mAP calculation using Torchmetrics.
            for i in range(len(images)):
                true_dict = dict()
                preds_dict = dict()
                true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
                true_dict['labels'] = targets[i]['labels'].detach().cpu()
                preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
                preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
                preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
                preds.append(preds_dict)
                preds2.append(outputs[i]['labels'][0].detach().cpu().numpy())
                gt.append(true_dict)

        self.mean_avg_precision(preds, gt)

        target = torch.tensor([int(item['labels']) for item in gt])
        predictions = torch.tensor([int(arr) for arr in preds2])

        self.precision(target, predictions)
        self.recall(target, predictions)
        self.f1_score(target, predictions)


if __name__ == '__main__':
    evaluation = EvalObjectDetectionModel()
    evaluation.validate()
