import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import datasets
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.config_parser import get_module
from light_trainer import trainer, recorder

import argparse
import importlib


import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True

import pdb


def main(args, config):
    # parsing cfg
    pGen, pDataset, pModel = config.get_config()

    prefix = pGen.name
    trainer.init_env(seed=5120)
    if args.test == False:
        # define dataloader
        val_dataset = get_module(type=pDataset.Val.type, config=pDataset.Val)
        val_sampler = DistributedSampler(val_dataset)
        loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=(val_sampler is None),
                                num_workers=pDataset.Val.num_workers,
                                sampler=val_sampler,
                                pin_memory=True,
                                prefetch_factor=2,
                                persistent_workers=True)
    else:
        # define dataloader
        test_dataset = get_module(type=pDataset.Test.type, config=pDataset.Test)
        test_sampler = DistributedSampler(test_dataset)
        loader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=(test_sampler is None),
                                num_workers=pDataset.Test.num_workers,
                                sampler=test_sampler,
                                pin_memory=True,
                                prefetch_factor=2,
                                persistent_workers=True)
    
    # define model
    model = get_module(type=pModel.type, pModel=pModel)
    model_trainer = eval(pModel.runner_type)(
        precision=args.precision,
        pModel=pModel,
        sync_batchnorm=False,
        test_save_path=os.path.splitext(args.resume_ckpt)[0]
    )
    model_trainer.predict(model, dataloader=loader, ckpt_path=args.resume_ckpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lidar panoptic segmentation')
    parser.add_argument('--config', help='config file path', type=str)
    parser.add_argument('--precision', help='precision of the float number', type=str, default="fp32")

    parser.add_argument('--resume_ckpt', help='resume checkpoint', type=str, default=None)
    parser.add_argument('--test', help='use test set', default=False, required=False, action='store_true')

    args = parser.parse_args()
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    main(args, config)