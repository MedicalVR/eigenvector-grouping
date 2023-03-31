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
    input_dir: str = os.path.join(".data", "input(artery_only)") # artery_only, normtot
    # input_dir: str = "D:\eigenvector-grouping\.data\input(artery_only)" 
    # input_dir: str = os.path.join(".data", "input(vein_only)") # veins_only, normtot
    # input_dir: str = os.path.join(".data", "input(artery_vein)") # artery_vein, normtot
    output_dir: str = os.path.join(".data", "output")
    

    # Data
    size: int = 20_000 # number of point selected in PointCloud, bijv. 95% of points
    split: str = None

    # Dataloader
    batch_size: int = 2
    num_workers: int = 1 # 10 # Number of subprocesses to use for data loading. 0 means 
    # that the data will be loaded in the main process. Number of CPUs available. 

    # Model
    model: object = PointNet2EVG
    model_type: str = "radius"
    ncomponents: int = 0 
    features: int = 0 # I don't know what this variable does 
    classes: int = 2 # 3 # do I need to change this if I'm changing input data? I did 

    # Trainer
    epochs: int = 20 # 200

    # Optimizer
    optimizer: object = OptimizerConfig(
        hyperparams=OptimizerHyperparamsConfig(lr=0.0025)
    )