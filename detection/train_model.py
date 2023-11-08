import colorama
import logging
import numpy as np
import os
import torch

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader import CaltechDatasetLoader
from detection_models.network_selectory import NetworkFactory
from settings.config import ConfigObjectDetection
from settings.dataset_network_configs import dataset_configs, network_configs
from utils.utils import collate_fn, get_train_transform, get_valid_transform, create_timestamp


class TrainObjectDetectionModel:
    def __init__(self) -> None:
        # Create time stamp
        self.timestamp = create_timestamp()

        colorama.init()

        self.cfg = ConfigObjectDetection().parse()

        dataset_config = dataset_configs(self.cfg)
        network_config = network_configs(self.cfg)

        train_dataset = CaltechDatasetLoader(dataset_config.get("train_images"),
                                             dataset_config.get("width"),
                                             dataset_config.get("height"),
                                             dataset_config.get("classes"),
                                             get_train_transform())

        valid_dataset = CaltechDatasetLoader(dataset_config.get("valid_images"),
                                             dataset_config.get("width"),
                                             dataset_config.get("height"),
                                             dataset_config.get("classes"),
                                             get_valid_transform())

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = NetworkFactory().create_network(self.cfg.type_of_net, dataset_config, device=self.device)

        parameters = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = \
            torch.optim.SGD(params=parameters,
                            lr=self.cfg.learning_rate,
                            momentum=self.cfg.momentum,
                            weight_decay=self.cfg.weight_decay)

        # LR scheduler
        self.scheduler = StepLR(optimizer=self.optimizer,
                                step_size=self.cfg.step_size,
                                gamma=self.cfg.gamma)

        # Tensorboard
        tensorboard_log_dir = os.path.join(network_config.get("logs_folder"), self.timestamp)
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

        # Create save directory
        self.save_path = os.path.join(network_config.get('weights_folder'), self.timestamp)
        os.makedirs(self.save_path, exist_ok=True)

    # --------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------T R A I N -------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------------- #
    def train(self):
        train_losses = []
        valid_losses = []

        # Variables to save only the best weights and model
        best_valid_loss = float('inf')
        best_model_path = None

        for epoch in tqdm(range(self.cfg.num_epochs), desc=colorama.Fore.GREEN + "Epochs"):
            self.model.train()
            for _, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                                desc=colorama.Fore.CYAN + "Train"):
                self.optimizer.zero_grad()

                images, targets = data

                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                losses.backward()
                self.optimizer.step()

                train_losses.append(losses.item())

            with torch.no_grad():
                for _, data in tqdm(enumerate(self.valid_loader), total=len(self.valid_loader),
                                    desc=colorama.Fore.MAGENTA + "Validation"):
                    images, targets = data
                    images = list(image.to(self.device) for image in images)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                    with torch.no_grad():
                        loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                    valid_losses.append(losses.item())

            # Decay the learning rate
            self.scheduler.step()

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)

            self.writer.add_scalars("Loss", {"train": train_loss, "validation": valid_loss}, epoch)

            logging.info(f'train_loss: {train_loss:.5f} valid_loss: {valid_loss:.5f}')

            train_losses.clear()
            valid_losses.clear()

            # Save the model and weights
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if best_model_path is not None:
                    os.remove(best_model_path)
                best_model_path = os.path.join(self.save_path, "epoch_" + str(epoch) + ".pt")
                torch.save(self.model.state_dict(), best_model_path)
                logging.info(f"New weights have been saved at epoch {epoch} with value of {valid_loss:.5f}")
            else:
                logging.warning(f"No new weights have been saved. Best valid loss was {best_valid_loss:.5f},\n "
                                f"current valid loss is {valid_loss:.5f}")

        # Close and flush SummaryWriter
        self.writer.close()
        self.writer.flush()


if __name__ == "__main__":
    try:
        faster_r_cnn = TrainObjectDetectionModel()
        faster_r_cnn.train()
    except KeyboardInterrupt as kie:
        logging.error(kie)
