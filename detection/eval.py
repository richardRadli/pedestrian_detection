import colorama
import gc
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import time
import torch
import torch.nn.utils.prune as prune
import typing

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

from dataloader import CaltechDatasetLoader
from detection_models.network_selectory import NetworkFactory
from settings.config import ConfigObjectDetection
from settings.dataset_network_configs import dataset_configs, network_configs
from utils.utils import (calculate_average, collate_fn, create_timestamp, get_valid_transform,
                         find_latest_file_in_latest_directory)
from quantization import QuantizedFasterRCNN


class EvalObjectDetectionModel:
    def __init__(self):
        self.timestamp = create_timestamp()

        colorama.init()

        self.cfg = ConfigObjectDetection().parse()

        self.dataset_config = dataset_configs(self.cfg)
        self.network_config = network_configs(self.cfg)

        test_dataset = CaltechDatasetLoader(self.dataset_config.get("test_images"),
                                            self.dataset_config.get("width"),
                                            self.dataset_config.get("height"),
                                            self.dataset_config.get("classes"),
                                            get_valid_transform())

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = NetworkFactory().create_network(self.cfg.type_of_net, self.dataset_config, device=self.device)

        latest_model_file = find_latest_file_in_latest_directory(self.network_config.get("weights_folder"))
        checkpoint = torch.load(latest_model_file, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)

        if self.cfg.prune:
            self.model.apply(self.custom_pruning)
            logging.info(f"Model is pruned: {prune.is_pruned(self.model)}, with sparsity of {self.cfg.sparsity}")

        # self.model = QuantizedFasterRCNN(model)  # model is a Faster R-CNN model
        # self.model.load_state_dict(torch.load("D:\storage_for_ped_det/data/weights/faster_rcnn_quantized/2023-11-09_15-12-01/epoch_29.pt"))
        # self.model.to(self.device)

        self.overall_precision = None
        self.overall_recall = None
        self.mAP = None
        self.avg_time = None

        self.plot_save_path = os.path.join(self.network_config.get("plotting_folder"), self.timestamp)
        os.makedirs(self.plot_save_path, exist_ok=True)

    def custom_pruning(self, module):
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=self.cfg.sparsity)

    @staticmethod
    def mean_avg_precision(predictions: typing.List, gt: typing.List):
        metric = MeanAveragePrecision()
        metric.update(predictions, gt)
        mAP = metric.compute()

        logging.info(mAP)

    def save_metrics_to_txt(self):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        results_df = pd.DataFrame({
            'Precision': [self.overall_precision],
            'Recall': [self.overall_recall],
            'mAP': [self.mAP],
            'time': [self.avg_time]
        })

        results_df.to_csv(os.path.join(self.network_config['prediction_folder'],
                                       self.timestamp + "_metrics_prediction.txt"), sep='\t', index=True)

    def calculate_metrics(self, iou_threshold=0.5, confidence_threshold=0.5, plot_results=False):
        self.model.eval()

        # Initialize tqdm progress bar.
        prog_bar = tqdm(self.test_loader, total=len(self.test_loader))
        all_true_positives = 0
        all_false_positives = 0
        all_actual_positives = 0
        gt = []
        preds = []
        elapsed_time = []

        for batch_idx, data in enumerate(prog_bar):
            images, targets = data
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                start = time.time()
                outputs = self.model(images, targets)
                end = time.time() - start
                elapsed_time.append(end)

            for i in range(len(images)):
                true_boxes = targets[i]['boxes'].cpu()
                pred_boxes = outputs[i]['boxes'].cpu()
                pred_scores = outputs[i]['scores'].cpu()
                iou_matrix = box_iou(true_boxes, pred_boxes)

                # Apply confidence threshold to filter predictions
                confident_preds = pred_scores >= confidence_threshold
                pred_boxes = pred_boxes[confident_preds]
                iou_matrix = iou_matrix[:, confident_preds]

                # Calculate true positives, false positives, and actual positives for each image
                image_true_positives = 0
                image_actual_positives = len(true_boxes)

                for _ in range(len(true_boxes)):
                    if iou_matrix.numel() == 0:
                        break
                    iou = iou_matrix.max(dim=1)[0]  # Maximum IoU for each true box
                    max_iou_value, max_iou_index = iou.max(dim=0)
                    if max_iou_value >= iou_threshold:
                        image_true_positives += 1
                        iou_matrix[max_iou_index, :] = -1  # Mark the used prediction as invalid

                image_false_positives = len(pred_boxes) - image_true_positives

                all_true_positives += image_true_positives
                all_actual_positives += image_actual_positives
                all_false_positives += image_false_positives

                # For mAP calculation using Torchmetrics.
                true_dict = dict()
                preds_dict = dict()

                true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
                true_dict['labels'] = targets[i]['labels'].detach().cpu()

                preds_dict['boxes'] = pred_boxes
                preds_dict['scores'] = pred_scores[confident_preds]
                preds_dict['labels'] = outputs[i]['labels'][confident_preds].cpu()

                preds.append(preds_dict)
                gt.append(true_dict)

                if plot_results:
                    iou_threshold_met = (image_true_positives / image_actual_positives) >= iou_threshold
                    if iou_threshold_met:
                        self.plot_predicted_boxes(images, targets, outputs, batch_idx, self.plot_save_path)

        self.avg_time = calculate_average(elapsed_time)
        logging.info(f"Average prediction time: {round(self.avg_time * 1000, 2)} ms")

        self.overall_precision = all_true_positives / (all_true_positives + all_false_positives)
        self.overall_recall = all_true_positives / all_actual_positives

        logging.info(f"Precision: {self.overall_precision}, Recall: {self.overall_recall}")

        # Calculate mAP using Torchmetrics
        logging.info("Calculating mAP")
        metric = MeanAveragePrecision()
        metric.update(preds, gt)
        self.mAP = metric.compute()

        logging.info(f"mAP: {self.mAP}")

        self.save_metrics_to_txt()

    @staticmethod
    def plot_predicted_boxes(images, targets, outputs, idx, save_path=None):
        for i, (image, output, target) in enumerate(zip(images, outputs, targets)):
            img = image.cpu().permute(1, 2, 0).numpy()

            fig, ax = plt.subplots(1)
            ax.imshow(img)

            for box, label in zip(target['boxes'].cpu(), output['labels'].cpu()):
                x, y, x_max, y_max = box
                rect = patches.Rectangle((x, y), x_max - x, y_max - y, linewidth=1, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
                ax.annotate(f'Label: {label}', (x, y), color='r')

            for box, score, label in zip(output['boxes'].cpu(), output['scores'].cpu(), output['labels'].cpu()):
                x, y, x_max, y_max = box
                rect = patches.Rectangle((x, y), x_max - x, y_max - y, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.annotate(f'Label: {label}, Score: {score:.2f}', (x, y), color='r')

            plt.axis('off')
            plt.title(f'Predicted Bounding Boxes - Image {idx}_{i}')

            if save_path:
                plt.savefig(f'{save_path}/image_{idx}_{i}.png')
                plt.close("all")
                gc.collect()
            else:
                plt.show()


if __name__ == '__main__':
    try:
        evaluation = EvalObjectDetectionModel()
        evaluation.calculate_metrics()
    except KeyboardInterrupt as kie:
        logging.error(f"Exception happened: {kie}")
