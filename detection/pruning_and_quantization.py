import torch
import torch.nn.utils.prune as prune

from detection_models.network_selectory import NetworkFactory
from settings.config import ConfigObjectDetection
from settings.dataset_network_configs import dataset_configs, network_configs
from utils.utils import find_latest_file_in_latest_directory


def prune_certain_layers(module, desired_sparsity: float = 0.5):
    prune.l1_unstructured(
        module, name="weight", amount=desired_sparsity
    )


def quantization(model):
    return torch.quantization.quantize_dynamic(
        model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
    )


def main():
    cfg = ConfigObjectDetection().parse()

    dataset_config = dataset_configs(cfg)
    network_config = network_configs(cfg)

    latest_model_file = find_latest_file_in_latest_directory(network_config.get("weights_folder"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NetworkFactory().create_network(cfg.type_of_net, dataset_config, device=device)

    checkpoint = torch.load(latest_model_file, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)

    desired_sparsity = 0.5  # 50% sparsity
    modules_to_prune = [
        model.model.backbone.body.conv1,
        model.model.backbone.body.layer1[0].conv1
    ]

    for module in modules_to_prune:
        prune_certain_layers(module, desired_sparsity)

    quantized_model = quantization(model)

    pruned_model_checkpoint = "C:/Users/ricsi/Desktop/pruned_model.pth"
    torch.save(quantized_model, pruned_model_checkpoint)

    # Load the pruned model checkpoint
    checkpoint = torch.load(pruned_model_checkpoint, map_location=device)

    print(checkpoint)


if __name__ == "__main__":
    main()
