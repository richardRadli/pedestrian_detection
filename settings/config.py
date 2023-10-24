import argparse


class ConfigObjectDetection:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--dataset_type", type=str, default="caltech", help="caltech | ecp")
        self.parser.add_argument("--type_of_net", type=str, default="SSD", help="Faster_R_CNN | SSD")
        self.parser.add_argument('--batch_size', type=int, default=4)
        self.parser.add_argument('--num_epochs', type=int, default=100)
        self.parser.add_argument('--learning_rate', type=float, default=2e-4) # 0.001
        self.parser.add_argument('--momentum', type=float, default=0.9)
        self.parser.add_argument('--weight_decay', type=float, default=0.0005)

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
