import logging

from typing import Dict
from settings.const import DATA_PATH, DATASET_PATH, IMAGES_PATH


def dataset_configs(cfg) -> Dict:
    dataset_type = cfg.dataset_type
    logging.info(dataset_type)

    dataset_config = {
        'caltech': {
            'train_images': DATASET_PATH.get_data_path("caltech_train"),
            'valid_images': DATASET_PATH.get_data_path("caltech_valid"),
            'test_images': DATASET_PATH.get_data_path("caltech_test"),
            'classes': ['background', 'Arduino_Nano', 'ESP8266', 'Raspberry_Pi_3', 'Heltec_ESP32_Lora'],
            'num_classes': 5,
            "width": 640,
            "height": 480,
            "nms": 0.45

        },
        "ecp": {
            'train_images': DATASET_PATH.get_data_path("ecp_train"),
            'valid_images': DATASET_PATH.get_data_path("ecp_valid"),
            'test_images': DATASET_PATH.get_data_path("ecp_test"),
            'classes': ['pedestrian'],
            'num_classes': 1
        }
    }

    if dataset_type not in dataset_config:
        raise ValueError(f'Invalid dataset type: {dataset_type}')

    return dataset_config[dataset_type]


def network_configs(cfg) -> Dict:
    """
    Returns a dictionary containing the prediction, plotting, and reference vectors folder paths for different types of
    networks based on the type_of_net parameter in cfg.
    :return: Dictionary containing the prediction and plotting folder paths.
    """

    network_type = cfg.type_of_net
    logging.info(network_type)
    detection_network_configs = {
        'Faster_R_CNN': {
            'logs_folder':
                DATA_PATH.get_data_path("logs_faster_rcnn"),
            'weights_folder':
                DATA_PATH.get_data_path("weights_faster_rcnn"),
            'prediction_folder':
                DATA_PATH.get_data_path("prediction_txt_faster_rcnn"),
            'plotting_folder':
                IMAGES_PATH.get_data_path("plotting_faster_rcnn"),
            'confusion_matrix':
                IMAGES_PATH.get_data_path("confusion_matrix_faster_rcnn"),
        },
        'SSD': {
            'logs_folder':
                DATA_PATH.get_data_path("logs_ssd"),
            'weights_folder':
                DATA_PATH.get_data_path("weights_ssd"),
            'prediction_folder':
                DATA_PATH.get_data_path("prediction_txt_ssd"),
            'plotting_folder':
                IMAGES_PATH.get_data_path("plotting_ssd"),
            'confusion_matrix':
                IMAGES_PATH.get_data_path("confusion_matrix_ssd"),
        },
    }

    if network_type not in detection_network_configs:
        raise ValueError(f'Invalid network type: {network_type}')

    return detection_network_configs[network_type]
