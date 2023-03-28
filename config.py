import os

from dataclasses import dataclass
from torch.optim import Adam

from src.models.pointnet2_msg import PointNet2MSG
from src.models.pointnet2_gag import PointNet2GAG
from src.models.pointnet2_evg import PointNet2EVG


@dataclass
class OptimizerHyperparamsConfig:
    lr: float = 0.0005
    weight_decay: float = 0
    betas: tuple = (0.9, 0.999)
    amsgrad: bool = False


@dataclass
class OptimizerConfig:
    optim: callable = Adam
    hyperparams: object = OptimizerHyperparamsConfig()

@dataclass
class TrainSegmentatorConfig:

    seed: int = 0

    # Paths
    # input_dir: str = os.path.join(".data", "input") # artery_vein, Laurens, normtot is used
    # input_dir: str = os.path.join(".data", "input(artery_only)") # artery_only, normtot
    input_dir: str = "D:\eigenvector-grouping\.data\input(artery_only)" 
    # input_dir: str = os.path.join(".data", "input(vein_only)") # veins_only, normtot
    # input_dir: str = os.path.join(".data", "input(artery_vein)") # artery_vein, normtot
    output_dir: str = os.path.join(".data", "output")
    

    # Data
    size: int = 20_000
    split: str = None

    # Dataloader
    batch_size: int = 2
    num_workers: int = 10

    # Model
    model: object = PointNet2EVG
    model_type: str = "radius"
    ncomponents: int = 0
    features: int = 0
    classes: int = 3 # 3 # do I need to change this if I'm changing input data? I did 

    # Trainer
    epochs: int = 50 # 200

    # Optimizer
    optimizer: object = OptimizerConfig(
        hyperparams=OptimizerHyperparamsConfig(lr=0.0025)
    )