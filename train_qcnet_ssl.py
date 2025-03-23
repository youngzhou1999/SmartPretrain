# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from datamodules import ArgoverseV2DataModule
from predictors import QCNet_ssl as QCNet
import os
import random
import torch
import numpy as np

########### train.py for self-supervised pretraining #############
# 1. Set different seed each epoch (for data augmentation randomness, etc.)
# 2. Add num_nodes argument for multi-node training
# 3. Use DDP strategy with sync_batchnorm


def build_worker_init_fn(base_seed):
    def worker_init_fn(worker_id):
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

        print(f"[Rank {os.environ.get('RANK', '?')}] Worker {worker_id} initialized with seed {worker_seed}")

    return worker_init_fn

class SeedCallback(pl.Callback):
    """
    Set a unique seed per GPU (global_rank) and per epoch.
    Also prepare a worker_init_fn that ensures different seed for each DataLoader worker.
    """
    def __init__(self, base_seed: int):
        self.base_seed = base_seed
        self.rank_seed = None

    def on_fit_start(self, trainer, pl_module):
        # Each GPU gets a unique base seed (global_rank-aware)
        self.rank_seed = self.base_seed + trainer.global_rank * 13579
        print(f"[Rank {trainer.global_rank}] base seed set to {self.rank_seed}")

        # Set base seed for current process
        random.seed(self.rank_seed)
        np.random.seed(self.rank_seed)
        torch.manual_seed(self.rank_seed)
        torch.cuda.manual_seed_all(self.rank_seed)

        # Save worker_init_fn to pl_module for later use
        pl_module.worker_init_fn = build_worker_init_fn(self.rank_seed)
        assert trainer.lightning_module.worker_init_fn is not None

    def on_epoch_start(self, trainer, pl_module):
        # Optionally change seed every epoch to introduce extra randomness
        epoch_seed = self.rank_seed + trainer.current_epoch * 77
        random.seed(epoch_seed)
        np.random.seed(epoch_seed)
        torch.manual_seed(epoch_seed)
        torch.cuda.manual_seed_all(epoch_seed)

        print(f"[Rank {trainer.global_rank}] Epoch {trainer.current_epoch} - seed set to {epoch_seed}")


if __name__ == '__main__':

    # -------- Argument Parser -------- #
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--is_pretrain', type=int, default=1)
    parser.add_argument('--resume_path', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)

    # Raw & processed data paths
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)

    # Lightning trainer config
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--num_nodes', type=int, default=1)

    # Dataset choices
    parser.add_argument('--load_av1', type=int, default=0)
    parser.add_argument('--load_waymo', type=int, default=0)

    # Add model-specific arguments from QCNet class
    QCNet.add_model_specific_args(parser)

    args = parser.parse_args()

    # -------- Trainer Setup -------- #
    args.devices = 'auto'               # Auto-select available devices
    args.accelerator = 'gpu'            # Use GPU
    args.strategy = "ddp_find_unused_parameters_true"  # Enable DDP and support unused params

    base_dir = "./logs_pretrain/"

    # Resume from checkpoint if provided
    if args.resume_path != 'null':
        rerun_ckpt = os.path.join(base_dir, args.resume_path, 'checkpoints', 'last.ckpt')
    else:
        rerun_ckpt = None

    # -------- Model & Data -------- #
    model = QCNet(**vars(args))

    # Select appropriate datamodule
    datamodule = {
        'argoverse_v2': ArgoverseV2DataModule,
    }[args.dataset](**vars(args))

    # Save top 5 checkpoints (by val_loss) and always save the last
    model_checkpoint = ModelCheckpoint(monitor='val_loss', save_last=True, save_top_k=5, mode='min')

    # Set seed callback to create different seeds across epochs/nodes
    seed_callback = SeedCallback(args.seed)

    # Monitor learning rate every epoch
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Initialize Lightning Trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        num_nodes=args.num_nodes,
        callbacks=[model_checkpoint, lr_monitor, seed_callback],
        max_epochs=args.max_epochs,
        default_root_dir=base_dir + args.exp_name,
        sync_batchnorm=True,  # Enable synchronized BatchNorm across GPUs
    )

    # -------- Start Training -------- #
    if rerun_ckpt is None:
        # Fresh training
        trainer.fit(model, datamodule)
    else:
        # Resume from previous checkpoint
        trainer.fit(model, datamodule, ckpt_path=rerun_ckpt)
