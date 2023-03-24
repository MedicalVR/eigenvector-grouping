import os
import datetime

import numpy as np
import pytorch_lightning
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from lightning_module import SegmentatorModule
from config import TrainSegmentatorConfig
 
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
# https://towardsdatascience.com/how-to-tune-pytorch-lightning-hyperparameters-80089a281646

def train_hp_loop(search_space):
    
    # Load config
    # change hyperparameters here 
    # config = TrainSegmentatorConfig(optimizer = OptimizerConfig(hyperparams=OptimizerHyperparamsConfig(lr=0.0025)))
    config = TrainSegmentatorConfig()
    np.random.seed(config.seed)

    # # Init out directories
    # ct = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    # logs_path = os.path.join(config.output_dir, ct, "logs")
    # checkpoint_path = os.path.join(config.output_dir, ct, "checkpoint")

    # os.makedirs(logs_path, exist_ok=True)
    # os.makedirs(os.path.join(logs_path, "lightning_logs"), exist_ok=True)
    # os.makedirs(checkpoint_path, exist_ok=True)
    # print("\nLogging to:", logs_path)

    # Init model
    net = SegmentatorModule(config)

    # # Set up loggers and checkpoints
    # tb_logger = TensorBoardLogger(save_dir=str(logs_path))
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=checkpoint_path,
    #     monitor="val_loss",
    #     mode="min",
    #     save_last=True,
    # )

    # Initialise Lightning's trainer.
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"} # added
    trainer = pytorch_lightning.Trainer(
        accelerator="gpu",
        max_epochs=config.epochs,
        # logger=tb_logger, # commented out 
        # callbacks=[checkpoint_callback], # commented out 
        num_sanity_val_steps=1,
        callbacks=[TuneReportCallback(metrics, on="validation_end")],    # added 
    )

    # Fit model
    trainer.fit(net)

    # best_model_path = checkpoint_callback.best_model_path
    # print(f"Best model path: {best_model_path}")
    # return best_model_path



def train():
    """Train surface model"""

    # data_dir = os.path.join(tempfile.gettempdir(), "mnist_data_")
    # # Download data
    # MNISTDataModule(data_dir=data_dir).prepare_data()

    # define search space for hyperparameters 
    search_space = {
        "lr": tune.uniform([0.0005, 0.005]),
        "weight_decay": tune.choice([0, 0.001, 0.005, 0.01]),
        "betas": tune.choice[(0.9, 0.999)],
        "amsgrad": tune.choice(['True','False'])
        }

    trainable = tune.with_parameters(
        train_hp_loop,
        # data_dir=data_dir, 
        # num_epochs=config.epochs,
        # num_gpus=gpus_per_trial
        )

    analysis = tune.run(
        trainable,
        resources_per_trial={
            "cpu": 1,
            "gpu": gpus_per_trial
            # cuda? 
        },
        metric="loss",
        mode="min",
        config=search_space,
        num_samples=num_samples,
        name="tune_mnist")

    print(analysis.best_config)
    
    # Load config
    # change hyperparameters here 
    config = TrainSegmentatorConfig(optimizer = OptimizerConfig(hyperparams=OptimizerHyperparamsConfig(lr=0.0025)) )
    np.random.seed(config.seed)

    # Init out directories
    ct = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    logs_path = os.path.join(config.output_dir, ct, "logs")
    checkpoint_path = os.path.join(config.output_dir, ct, "checkpoint")

    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(os.path.join(logs_path, "lightning_logs"), exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    print("\nLogging to:", logs_path)

    # Init model
    net = SegmentatorModule(config)

    # Set up loggers and checkpoints
    tb_logger = TensorBoardLogger(save_dir=str(logs_path))
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    # Initialise Lightning's trainer.
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"} # added
    callbacks = [TuneReportCallback(metrics, on="validation_end")] # added
    trainer = pytorch_lightning.Trainer(
        accelerator="gpu",
        max_epochs=200,
        logger=tb_logger,
        # callbacks=[checkpoint_callback], # commented out 
        num_sanity_val_steps=1,
        callbacks=callbacks,    # added 
    )

    # Fit model
    trainer.fit(net)

    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model path: {best_model_path}")
    return best_model_path


if __name__ == "__main__":
    # search_space = {
    # "layer_1_size": tune.choice([32, 64, 128]),
    # "layer_2_size": tune.choice([64, 128, 256]),
    # "lr": tune.loguniform(1e-4, 1e-1),
    # "batch_size": tune.choice([32, 64, 128])
    # }
    search_space = {
    "learning_rate": tune.uniform([0.0005, 0.005,]),
    "weight_decay": tune.choice([0, 0.001, 0.005, 0.01]),
    "betas": tune.choice[(0.9, 0.999)],
    "amsgrad": tune.choice(['True','False'])
    }
    lr: float = 0.0005
    weight_decay: float = 0
    betas: tuple = (0.9, 0.999)
 
    train()
