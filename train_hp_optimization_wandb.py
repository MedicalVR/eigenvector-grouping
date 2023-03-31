# add self.save_hyperparameters in lightning_module.py
# remove absolute path and add relative path in config.py
# num-workers in config.py

import os
import datetime
import wandb

import numpy as np
import pytorch_lightning
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from lightning_module import SegmentatorModule
from config import TrainSegmentatorConfig


def train():
    """Train surface model"""
    
    # Load config default (from config.py)
    config_default = TrainSegmentatorConfig() # change hyperparameters in config.py here 
    np.random.seed(config_default.seed)

    # Init out directories
    ct = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    logs_path = os.path.join(config_default.output_dir, ct, "logs")
    checkpoint_path = os.path.join(config_default.output_dir, ct, "checkpoint")

    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(os.path.join(logs_path, "lightning_logs"), exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    print("\nLogging to:", logs_path)

    # # Init model
    # net = SegmentatorModule(config)

    # Set up loggers and checkpoints
    # tb_logger = TensorBoardLogger(save_dir=str(logs_path)) # co
    # wandb.init(project="hp-optizimation", config=config_default) # co
    # config = config_default 
    # config = wandb.config # co  
    wandb_logger = WandbLogger(project='hp-optimization')
    print(wandb.config['batch_size']) 
    print(wandb.config['epochs']) 
    hp_sweep = wandb.config
    print(hp_sweep) 
    dict_key_hp_sweep = list(hp_sweep.keys())
    # print(dict_key_config)

    # update hyperparameters in config_default, which will be used to construct the lightning module class
    # cave: did not put lr, weight_decay etc. in config.py so those cannot be updated at the moment 
    if 'model_type' in dict_key_hp_sweep:
        config_default.model_type = hp_sweep['model_type']
    if 'size' in dict_key_hp_sweep:
        config_default.size = hp_sweep['size']
    if 'split' in dict_key_hp_sweep:
        config_default.split = hp_sweep['split']
    if 'batch_size' in dict_key_hp_sweep:
        config_default.batch_size = hp_sweep['batch_size']
    if 'num_workers' in dict_key_hp_sweep:
        config_default.num_workers = hp_sweep['num_workers']
    if 'ncomponents' in dict_key_hp_sweep:
        config_default.ncomponents = hp_sweep['ncomponents']
    if 'features' in dict_key_hp_sweep:
        config_default.features = hp_sweep['features']
    if 'epochs' in dict_key_hp_sweep:
        config_default.epochs = hp_sweep['epochs']
    if 'optimizer' in dict_key_hp_sweep: 
        # if you want to use a different optimizer, you have to import it first in config.py
        config_default.optimizer.optim = hp_sweep['optimizer']
    if 'lr' in dict_key_hp_sweep:
        config_default.optimizer.hyperparams.lr = hp_sweep['lr']
    if 'weight_decay' in dict_key_hp_sweep:
        config_default.optimizer.hyperparams.weight_decay = hp_sweep['weight_decay']
    if 'beta1' in dict_key_hp_sweep and 'beta2' in dict_key_hp_sweep:
        config_default.optimizer.hyperparams.betas = (hp_sweep['beta1'],hp_sweep['beta2'])
    if 'amsgrad' in dict_key_hp_sweep:
        config_default.optimizer.hyperparams.amsgrad = bool(hp_sweep['amsgrad'])

    print(config_default)

    # wandb_logger = WandbLogger() # co 
    # wandb_logger.experiment.config["batch_size"] = config.batch_size # co
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    # Init model
    net = SegmentatorModule(config_default)

    # Initialise Lightning's trainer.
    trainer = pytorch_lightning.Trainer(
        accelerator="gpu",
        max_epochs=config_default.epochs, # 200, # changed because of time
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=1,
    )

    # Fit model
    trainer.fit(net)

    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model path: {best_model_path}")
    return best_model_path


if __name__ == "__main__":
    sweep_configuration = {
        'method': 'random',
        'metric': {'goal': 'minimize', 'name': 'val_loss'},
        'parameters': 
        {
            "size": {'min': 20_000, 'max': 30_000}, # 'max': 80_000}, # use these bigger values on faster computer
            "batch_size" : {'value': 2}, # {'values': [1,2,4]} # use these bigger values on faster computer
            # "model_type": {'values': ["radius", "knn"]}, # knn does not seem to work yet
            "epochs" : {'min': 3, 'max': 10}, # 10,200),
            "lr": {'min': 1e-7, 'max': 1e-1},
            "weight_decay": {"min": 0.0, 'max': 0.5}, # {'values': [0, 0.001, 0.005, 0.01]},
            "beta1": {'min': 0.5, 'max': 1.0},
            "beta2": {'min': 0.5, 'max': 1.0}, 
            "amsgrad": {"values": ['True','False']}, 
        }
    }  

    sweep_id = wandb.sweep(sweep_configuration, project='hp-optimization')
    wandb.agent(sweep_id=sweep_id, function=train, count=3)

    # train()