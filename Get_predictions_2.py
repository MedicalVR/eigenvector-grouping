
#%% import relevant modules 
import sys
import numpy as np
import torch
import glob # finds all the pathnames matching a specified pattern according to the rules used by the Unix shell, although results are returned in
# arbitrary order
import monai # build on top of pytorch
from monai.transforms import Compose, LoadImaged
from monai.data import list_data_collate
# I added these
import pytorch_lightning

#%% import files 
# insert path of modules folders 
# sys.path.insert(0,  'D:/eigenvector-grouping')

# import the modules directly
from lightning_module import SegmentatorModule
from src.transforms.transforms import (
    PointcloudRandomSubsampled,
    ExtractSegmentationLabeld,
    ToFloatTensord
)


#%% import data 
# change the directory for (vein_only) and (atery_vein)
files_root = glob.glob(r'D:\Data\LungSegmentations\PointClouds(artery_only)\PointClouds\Test_set\*')
test_dict = [{"input": file} for file in files_root[:2]] # only two files
# test_dict = [{"input": file} for file in files_root]

#%% try lightning example
# https://lightning.ai/forums/t/how-to-load-and-use-model-checkpoint-ckpt/677 
model = SegmentatorModule()
trainer = pytorch_lightning.Trainer()
chk_path = "D:/eigenvector-grouping/.data/output/01-20-2023-13-49-42/checkpoint/epoch=191-step=3264.ckpt"
model2 = SegmentatorModule.load_from_checkpoint(chk_path)
results = trainer.test(model=model2, datamodule=my_datamodule, verbose=True)
trainer = Trainer()
trainer.fit(model)

# automatically loads the best weights for you
trainer.test(model)


#%% try train loader etc. from lightning_module.py as it should have the save structure
# https://lightning.ai/docs/pytorch/stable/data/datamodule.html
test_transforms = Compose([
    LoadImaged(keys=["input"], reader="NumpyReader"),
    PointcloudRandomSubsampled(keys=["input"], sub_size=20_000),
    ExtractSegmentationLabeld(pcd_key="input"),
    ToFloatTensord(keys=["input", "label"]),
])

test_dataset = monai.data.CacheDataset(
    data=test_dict,
    transform=test_transforms,
    cache_rate=1.0,
    num_workers=10,
)

test_loader = monai.data.DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=10,
    collate_fn=list_data_collate,
)

#%% other example thing lightning
chk_path = "D:/eigenvector-grouping/.data/output/01-20-2023-13-49-42/checkpoint/epoch=191-step=3264.ckpt"
model = SegmentatorModule.load_from_checkpoint(chk_path)
trainer = pytorch_lightning.Trainer()
trainer.test(model, dataloaders=test_loader)


#%% test transforms
# provides the ability to chain a series of callables together in a sequential manner. Each transform in the sequence must take a single argument and 
# return a single value.
test_transforms = Compose([
            LoadImaged(keys=["input"], reader="NumpyReader"),
            # PointcloudRandomSubsampled(keys=["input"], sub_size=20_000),
            ExtractSegmentationLabeld(pcd_key="input"),
            ToFloatTensord(keys=["input", "label"])
        ])


#%% test dataset
# Dataset with cache mechanism that can load data and cache deterministic transforms’ result during training
test_dataset = monai.data.CacheDataset(
            data=test_dict,
            transform=test_transforms,
            cache_rate=1.0,
            num_workers=1, # 10, the number of worker threads if computing cache in the initialization
        )



#%% test loader
# Provides an iterable over the given dataset
multiprocessing_context_name = 'fork'
test_loader = monai.data.DataLoader(
    test_dataset,
    batch_size=1,
    #shuffle=True,
    num_workers=10,
    collate_fn=list_data_collate,
    # multiprocessing_context=multiprocessing_context_name,
    )

#%% import model from checkpoint
# add different checkpoint_path probably
checkpoint_path = "D:/eigenvector-grouping/.data/output/01-20-2023-13-49-42/checkpoint/epoch=191-step=3264.ckpt"
model = SegmentatorModule.load_from_checkpoint(checkpoint_path)
# don't use .to(device) --> https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)


#%% evaluate model 
model.eval()
i=0
# model.to(device)
for batch_data in test_loader:
    input_tensor = batch_data["input"]
    # input_tensor.to(device)
    label_pred = model(input_tensor)
    i+=1
    #tensor=label_pred.detach().numpy()
    #tensor=tesnor[0,:,:,:].squeeze()



# %%
