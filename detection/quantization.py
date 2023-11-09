import os
import torch

from detection_models.network_selectory import NetworkFactory
from settings.config import ConfigObjectDetection
from settings.dataset_network_configs import dataset_configs, network_configs
from utils.utils import find_latest_file_in_latest_directory, create_timestamp
import torch.quantization


class QuantizedFasterRCNN(torch.nn.Module):
    def __init__(self, model):
        super(QuantizedFasterRCNN, self).__init__()
        self.model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

    def forward(self, x, y):
        return self.model(x, y)


def print_file_size(original, pruned):
    size_original = os.path.getsize(original)
    size_pruned = os.path.getsize(pruned)
    print(f"The size of the original file is {size_original / 10**6} MB")
    print(f"The size of the pruned file is {size_pruned / 10**6} MB")

    change = size_pruned - size_original
    percentage_change = (change / size_original) * 100

    if change > 0:
        print(f"The size of the pruned file increased by {percentage_change:.4f}% compared to the original file.")
    elif change < 0:
        print(f"The size of the pruned file decreased by {percentage_change:.4f}% compared to the original file.")
    else:
        print("No change has happened")


def main():
    timestamp = create_timestamp()

    cfg = ConfigObjectDetection().parse()

    dataset_config = dataset_configs(cfg)
    network_config = network_configs(cfg)

    latest_model_file = find_latest_file_in_latest_directory(network_config.get("weights_folder"))

    quantization_dir = os.path.join(network_config.get("quantized_weights_folder"), timestamp)
    os.makedirs(quantization_dir, exist_ok=True)
    quantized_model_checkpoint = os.path.join(quantization_dir, os.path.basename(latest_model_file))

    device = "cpu"
    model = NetworkFactory().create_network(cfg.type_of_net, dataset_config, device=device)
    checkpoint = torch.load(latest_model_file, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)

    total_parameters = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the quantized Faster R-CNN model: {total_parameters}")

    quantized_model = QuantizedFasterRCNN(model)
    torch.save(quantized_model.state_dict(), quantized_model_checkpoint)

    print_file_size(latest_model_file, quantized_model_checkpoint)

    # Loading procedure
    a = QuantizedFasterRCNN(model)  # model is a Faster R-CNN model
    a.load_state_dict(torch.load(quantized_model_checkpoint))


if __name__ == "__main__":
    main()
