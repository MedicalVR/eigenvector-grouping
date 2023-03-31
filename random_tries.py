from dataclasses import dataclass
from config import TrainSegmentatorConfig

@dataclass
class OptimizerHyperparamsConfig:
    lr: float = 0.0005
    weight_decay: float = 0
    betas: tuple = (0.9, 0.999)
    amsgrad: bool = False

print(OptimizerHyperparamsConfig.lr)

OptimizerHyperparamsConfig.lr = 0.0025

print(OptimizerHyperparamsConfig.lr)

# config
config = TrainSegmentatorConfig()

max_epoch = config.epochs
print(max_epoch)

config.epochs = 100
print(max_epoch)
print(config.epochs)


