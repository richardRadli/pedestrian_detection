import os
import torch
import torch.nn.utils.prune as prune

from detection_models.network_selectory import NetworkFactory
from settings.config import ConfigObjectDetection
from settings.dataset_network_configs import dataset_configs, network_configs
from utils.utils import find_latest_file_in_latest_directory


def prune_certain_layers(module, desired_sparsity: float = 0.5):
    prune.ln_structured(
        module, name="weight", amount=desired_sparsity, n=2, dim=0
    )


def save_quantized_model(model, pruned_model_checkpoint):
    quantized_model = torch.quantization.quantize_dynamic(
        model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
    )
    torch.save(quantized_model.state_dict(), pruned_model_checkpoint)


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
    pruned_model_checkpoint = "C:/Users/ricsi/Desktop/pruned_model.pt"

    cfg = ConfigObjectDetection().parse()

    dataset_config = dataset_configs(cfg)
    network_config = network_configs(cfg)

    latest_model_file = find_latest_file_in_latest_directory(network_config.get("weights_folder"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NetworkFactory().create_network(cfg.type_of_net, dataset_config, device=device)
    checkpoint = torch.load(latest_model_file, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)

    save_quantized_model(model, pruned_model_checkpoint)

    print_file_size(latest_model_file, pruned_model_checkpoint)

    quantized_model = NetworkFactory().create_network(cfg.type_of_net, dataset_config, device=device)
    quantized_model.load_state_dict(torch.load(pruned_model_checkpoint))


if __name__ == "__main__":
    main()
