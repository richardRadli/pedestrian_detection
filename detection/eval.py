import colorama
import cv2
import logging
import pandas as pd
import numpy as np
import os
import torch
import typing

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

from dataloader import CaltechDatasetLoader
from detection_models.network_selectory import NetworkFactory
from settings.config import ConfigObjectDetection
from settings.dataset_network_configs import dataset_configs, network_configs
from utils.utils import collate_fn, create_timestamp, get_valid_transform, find_latest_file_in_latest_directory


class EvalObjectDetectionModel:
    def __init__(self):
        self.timestamp = create_timestamp()

        colorama.init()

        self.cfg = ConfigObjectDetection().parse()

        self.dataset_config = dataset_configs(self.cfg)
        self.network_config = network_configs(self.cfg)

        latest_model_file = find_latest_file_in_latest_directory(self.network_config.get("weights_folder"))

        test_dataset = CaltechDatasetLoader(self.dataset_config.get("test_images"),
                                            self.dataset_config.get("width"),
                                            self.dataset_config.get("height"),
                                            self.dataset_config.get("classes"),
                                            get_valid_transform())

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = NetworkFactory().create_network(self.cfg.type_of_net, self.dataset_config, device=self.device)

        checkpoint = torch.load(latest_model_file, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        # latest_model_file = "C:/Users/ricsi/Desktop/pruned_model.pth"
        # self.model = torch.load(latest_model_file, map_location=self.device)
        self.overall_precision = None
        self.overall_recall = None
        self.mAP = None

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
            'mAP': [self.mAP]
        })

        results_df.to_csv(os.path.join(self.network_config['prediction_folder'],
                                       self.timestamp + "_metrics_prediction.txt"), sep='\t', index=True)

    def calculate_metrics(self, iou_threshold=0.5):
        self.model.eval()

        # Initialize tqdm progress bar.
        prog_bar = tqdm(self.test_loader, total=len(self.test_loader))
        all_true_positives = 0
        all_false_positives = 0
        all_actual_positives = 0
        gt = []
        preds = []

        for idx, data in enumerate(prog_bar):
            images, targets = data
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                outputs = self.model(images, targets)

            for i in range(len(images)):
                true_boxes = targets[i]['boxes'].cpu()
                pred_boxes = outputs[i]['boxes'].cpu()
                iou_matrix = box_iou(true_boxes, pred_boxes)

                # Calculate true positives, false positives, and actual positives for each image
                image_true_positives = 0
                image_actual_positives = len(true_boxes)

                for _, _ in zip(true_boxes, pred_boxes):
                    iou = iou_matrix.max(dim=1)[0]  # Maximum IoU for each true box
                    if iou.max() >= iou_threshold:
                        image_true_positives += 1

                image_false_positives = len(pred_boxes) - image_true_positives

                all_true_positives += image_true_positives
                all_actual_positives += image_actual_positives
                all_false_positives += image_false_positives

                # For mAP calculation using Torchmetrics.
                true_dict = dict()
                preds_dict = dict()

                true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
                true_dict['labels'] = targets[i]['labels'].detach().cpu()

                preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
                preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
                preds_dict['labels'] = outputs[i]['labels'].detach().cpu()

                preds.append(preds_dict)
                gt.append(true_dict)

            if idx == 10:
                break
            # for img, pred in zip(images, preds):
            #     self.plot_detected_bboxes([img], [pred], detection_threshold=0.1)

        self.overall_precision = all_true_positives / (all_true_positives + all_false_positives)
        self.overall_recall = all_true_positives / all_actual_positives

        logging.info(f"Precision: {self.overall_precision}, Recall: {self.overall_recall}")

        # Calculate mAP using Torchmetrics
        metric = MeanAveragePrecision()
        metric.update(preds, gt)
        self.mAP = metric.compute()

        logging.info(f"mAP: {self.mAP}")

        self.save_metrics_to_txt()

        return images, outputs

    def plot_detected_bboxes(self, images, outputs, detection_threshold: float = 0.1):
        for image, output in zip(images, outputs):
            tensor_image = image

            # Convert the tensor image to a NumPy array
            numpy_image = tensor_image.cpu().numpy()

            # Rearrange the color channels from RGB to BGR order
            bgr_image = numpy_image[[2, 1, 0], :, :]

            # Convert the NumPy array to a BGR image
            bgr_image = np.transpose(bgr_image, (1, 2, 0))
            bgr_image *= 255.0
            bgr_image = bgr_image.astype(np.uint8)

            # Create a copy of the BGR image to draw on
            image_with_boxes = bgr_image.copy()

            output = {k: v.to('cpu') for k, v in output.items()}

            if len(output['boxes']) != 0:
                boxes = output['boxes'].data.numpy()
                scores = output['scores'].data.numpy()
                boxes = boxes[scores >= detection_threshold].astype(np.int32)
                draw_boxes = boxes.copy()

                pred_classes = [self.dataset_config.get("classes")[i] for i in output['labels'].cpu().numpy()]

                for j, box in enumerate(draw_boxes):
                    cv2.rectangle(img=image_with_boxes,
                                  pt1=(int(box[0]), int(box[1])),
                                  pt2=(int(box[2]), int(box[3])),
                                  color=(0, 0, 255),
                                  )
                    cv2.putText(image_with_boxes, (pred_classes[j] + " " + str(scores[j])),
                                (int(box[0]), int(box[1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                                2, lineType=cv2.LINE_AA)

                cv2.imshow('pred', image_with_boxes)
                cv2.waitKey()


if __name__ == '__main__':
    evaluation = EvalObjectDetectionModel()
    images, outputs=evaluation.calculate_metrics()