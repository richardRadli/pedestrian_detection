import argparse


class ConfigObjectDetection:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--dataset_type", type=str, choices="caltech")
        self.parser.add_argument("--type_of_net", type=str, default="Faster_R_CNN_SA",
                                 choices="Faster_R_CNN | Faster_R_CNN_SA | SSD")
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--num_epochs', type=int, default=50)  # 40 for FRCNN
        self.parser.add_argument('--learning_rate', type=float, default=0.001)  # 0.001 for FRCNN
        self.parser.add_argument('--momentum', type=float, default=0.9)
        self.parser.add_argument('--weight_decay', type=float, default=1e-5)
        self.parser.add_argument('--prune', type=bool, default=False)
        self.parser.add_argument('--sparsity', type=float, default=0.2)
        self.parser.add_argument('--step_size', type=int, default=5,
                                 help="Number of epochs after which to decay the learning rate")
        self.parser.add_argument('--gamma', type=float, default=0.1, help="Factor by which to decay the learning rate")
        self.parser.add_argument('--quantized_model', type=bool, default=False)

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
