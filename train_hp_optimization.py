import os
import datetime

import numpy as np
import pytorch_lightning
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from lightning_module import SegmentatorModule
from config import TrainSegmentatorConfig
 
from ray import air, tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
# https://towardsdatascience.com/how-to-tune-pytorch-lightning-hyperparameters-80089a281646
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune import CLIReporter

import torch

# functions for the first try 
def train_hp_loop(search_space):
    
    # Load config
    # change hyperparameters here 
    # config = TrainSegmentatorConfig(optimizer = OptimizerConfig(hyperparams=OptimizerHyperparamsConfig(lr=0.0025)))
    config = TrainSegmentatorConfig()
    np.random.seed(config.seed)

    # # Init out directories
    ct = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    logs_path = os.path.join(config.output_dir, ct, "logs")
    checkpoint_path = os.path.join(config.output_dir, ct, "checkpoint")

    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(os.path.join(logs_path, "lightning_logs"), exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    print("\nLogging to:", logs_path)

    # Init model
    net = SegmentatorModule(config)

    # # Set up loggers and checkpoints
    tb_logger = TensorBoardLogger(save_dir=str(logs_path))
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    # Initialise Lightning's trainer.
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"} # added
    trainer = pytorch_lightning.Trainer(
        accelerator="auto", # "gpu", # I changed it to auto 
        max_epochs=config.epochs,
        logger=tb_logger,  
        # callbacks=[checkpoint_callback], # commented out 
        num_sanity_val_steps=1,
        # callbacks=[TuneReportCallback(metrics, on="validation_end")],    # added 
        callbacks=[checkpoint_callback, TuneReportCheckpointCallback(metrics, on="validation_end")],    # added
    )


    # Fit model
    trainer.fit(net)

    # best_model_path = checkpoint_callback.best_model_path
    # print(f"Best model path: {best_model_path}")
    # return best_model_path


def train(num_samples = 10, num_epochs = 200, gpus_per_trial = 0):
    """Train surface model"""

    # data_dir = os.path.join(tempfile.gettempdir(), "mnist_data_")
    # # Download data
    # MNISTDataModule(data_dir=data_dir).prepare_data()

    # define search space for hyperparameters 
    search_space = {
        "lr": tune.loguniform(1e-7, 1e-1),
        "weight_decay": tune.choice([0, 0.001, 0.005, 0.01]),
        # "betas": tune.choice((0.9, 0.999)), # how to give a tuple as input? 
        "amsgrad": tune.choice(['True','False']),
        "batch_size" : tune.choice([2]),
        "epochs" : tune.uniform(5,50) # 10-200
        }

    trainable = tune.with_parameters(
        train_hp_loop(search_space),
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
    
    best_model_path = analysis.best_checkpoint
    print(f"Best checkpoint: {best_model_path}")
    return best_model_path


# functions for the second try 
def train_tune(search_space, num_epochs=200, num_gpus=0):
    # https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html#putting-it-together

    config = TrainSegmentatorConfig()
    np.random.seed(config.seed) 

    # adapt config to the correct parameters of search_space

    model = SegmentatorModule(config)
    ct = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    # logs_path = os.path.join(config.output_dir, ct, "logs")
    logs_path = os.path.join(ct,"lightning_logs")
    # checkpoint_path = os.path.join(config.output_dir, ct, "checkpoint")
    checkpoint_path = os.path.join(ct, "checkpoint")
    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(os.path.join(logs_path, "lightning_logs"), exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    print("\nLogging to:", logs_path)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    metrics = {"loss": "ptl/val_loss", "mean_accuracy": "ptl/val_accuracy"}
    # default_root_dir=root_dir,
    #                  callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),tune_report_callback]
    tune_report_callback = TuneReportCheckpointCallback(
        metrics,
        filename="ray_ckpt",
        on="validation_end",
        )

    trainer = pytorch_lightning.Trainer(
        max_epochs=config.epochs,
        # If fractional GPUs passed in, convert to int.
        # gpus= math.ceil(num_gpus), # commented out
        accelerator="auto", #  "gpu", # I changed it to auto 
        logger=TensorBoardLogger(save_dir=str(logs_path)),
        enable_progress_bar=False,
        # callbacks=[
        #     TuneReportCallback(metrics,on="validation_end")
        #     # TuneReportCheckpointCallback(
        #     #     metrics,
        #     #     filename="ray_ckpt",
        #     #     on="validation_end")
        # ] 
        # callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),tune_report_callback]
        callbacks=[checkpoint_callback,tune_report_callback]
    )
    trainer.fit(model)

    print("------------------ Model trained ------------------")

def train_ASHA(num_samples = 10, num_epochs = 200, gpus_per_trial = 0):
    # define search space for hyperparameters 
    search_space = {
        "lr": tune.loguniform(1e-7, 1e-1),
        "weight_decay": tune.choice([0, 0.001, 0.005, 0.01]),
        # "betas": tune.choice(tuple(0.9, 0.999)), # how to input tuple to tune.choice
        "amsgrad": tune.choice(['True','False']),
        "batch_size" : tune.choice([2]),
        "epochs" : tune.randint(5, 50) # 10,200)
        }
    
    # maybe also try Population Based Training 
    # look into this with other values of parameters 
    scheduler = ASHAScheduler(
        max_t=num_epochs, # max time units per trial. Trials will be stopped after
            # max_t time units (determined by time_attr) have passed.
        grace_period=1, # Only stop trials at least this old in time.
        reduction_factor=2 # Used to set halving rate and amount
        )
    
    train_with_parameters = tune.with_parameters(train_tune, num_epochs = num_epochs, num_gpus = gpus_per_trial)

    resources_per_trial = {"cpu": 1, "gpu": gpus_per_trial}
    reporter = CLIReporter(
        parameter_columns=["lr", "weight_decay", "amsgrad", "batch_size", "epochs"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    tuner = tune.Tuner(
        tune.with_resources(
            train_with_parameters,
            resources=resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            metric="val_loss", # chane back to "loss"
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            name="tune_asha",
            local_dir=TrainSegmentatorConfig().output_dir,
            progress_reporter=reporter, # import CLIreporter for this
        ),
        param_space=search_space,
    )
    # tuner.report()
    results = tuner.fit()

    print('insert results.get_best_results().config here')
    # if results.get_best_result().config is None: 
    #     print('No best trial found for the given metric: loss. This means that no trial has reported this metric or all values are NaN') 
    # else: 
    #     print("Best hyperparameters found were: ", results.get_best_result().config)   

# run the train functions (only the first OR the second try)
if __name__ == "__main__":
    
    num_samples = 2 # 10 # how many trials are performed 
    gpus_per_trial = 1 if torch.cuda.is_available() else 0
    num_epochs = 200 # max, otherwise it takes too long

    # first try 
    # train(num_samples, gpus_per_trial, num_epochs)

    # second try 
    train_ASHA(num_samples, num_epochs, gpus_per_trial)

    
