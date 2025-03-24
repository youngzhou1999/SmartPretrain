# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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
from predictors import QCNet

from collections import OrderedDict
import os

def remove_prefix_if_needed(weight_dict, key, copy_refine=1):
    """
    Processes a weight dictionary by removing 'momentum_branch' and submodule
    prefixes, adjusts specific embeddings, and optionally copies weights for refinement layers.

    Args:
        weight_dict (dict): The original state_dict from a pretrained model.
        key (str): Submodule name to extract from (e.g., 'encoder' or 'decoder').
        copy_refine (int): If 1, also copy propose attention layers to corresponding refine layers.

    Returns:
        new_weight_dict (OrderedDict): A new dictionary with cleaned keys and processed values.
    """

    new_weight_dict = OrderedDict()  # Resulting cleaned weights

    for k, v in weight_dict.items():
        # Process only keys that start with 'momentum_branch'
        if k.startswith('momentum_branch'):
            # Strip the 'momentum_branch.' prefix
            new_k = k.replace('momentum_branch' + '.', '', 1)

            # Keep only entries that start with the given module key
            if new_k.startswith(key):
                # Skip processing if the key is 'decoder' (reserved for specific behavior)
                if key == 'decoder':
                    pass

                # Remove the second prefix (e.g., 'encoder.')
                k_ = new_k.replace(key + '.', '', 1)

                # Special case: expand mode embedding to 6 repeated values
                if 'mode_emb' in k_:
                    if 'weight' in k_:
                        v = v.repeat(6, 1)  # Repeat rows
                    else:
                        v = v.repeat(6)     # Repeat vector

                # Truncate certain embeddings to fixed lengths (dataset specific)
                if 'type_pt_emb' in k_:
                    v = v[:17]
                if 'side_pt_emb' in k_:
                    v = v[:3]
                if 'type_pl_emb' in k_:
                    v = v[:4]
                if 'type_a_emb' in k_:
                    v = v[:10]

                # Save the cleaned key and adjusted value
                new_weight_dict[k_] = v

                # If enabled, also copy propose attention layer weights to refine layer names
                if copy_refine:
                    if 't2m_propose_attn_layers' in k_:
                        k__ = k_.replace('t2m_propose_attn_layers', 't2m_refine_attn_layers')
                        new_weight_dict[k__] = v

                    if 'pl2m_propose_attn_layers' in k_:
                        k__ = k_.replace('pl2m_propose_attn_layers', 'pl2m_refine_attn_layers')
                        new_weight_dict[k__] = v

                    if 'a2m_propose_attn_layers' in k_:
                        k__ = k_.replace('a2m_propose_attn_layers', 'a2m_refine_attn_layers')
                        new_weight_dict[k__] = v

                    if 'm2m_propose_attn_layer' in k_:
                        k__ = k_.replace('m2m_propose_attn_layer', 'm2m_refine_attn_layer')
                        new_weight_dict[k__] = v
        else:
            continue  # Ignore all other keys

    return new_weight_dict


if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, required=True)
    parser.add_argument('--max_epochs', type=int, default=64)
    
    # for load pretrained model
    parser.add_argument('--pretrained_ckpt', type=str, default=None)
    parser.add_argument('--copy_refine', type=int, default=0)

    QCNet.add_model_specific_args(parser)
    args = parser.parse_args()

    model = QCNet(**vars(args))

    # Check if a pretrained checkpoint path is provided
    if args.pretrained_ckpt is not None:
        print(f'fine-tuning: {args.pretrained_ckpt}')  # Log which checkpoint is being used

        import torch
        # Load the checkpoint file, and extract the 'state_dict' (contains model weights)
        pre_trained_state_dict = torch.load(args.pretrained_ckpt)['state_dict']

        # Load encoder weights from the pretrained checkpoint
        # Uses `remove_prefix_if_needed` to clean keys and optionally copy refine layers
        # `strict=True` means all weights in the encoder must match exactly
        model.encoder.load_state_dict(
            remove_prefix_if_needed(pre_trained_state_dict, 'encoder', args.copy_refine),
            strict=True
        )

        # Load decoder weights from the pretrained checkpoint
        # `strict=False` allows partial loading (e.g., skipping some missing or extra keys)
        model.decoder.load_state_dict(
            remove_prefix_if_needed(pre_trained_state_dict, 'decoder', args.copy_refine),
            strict=False
        )
    
    datamodule = {
        'argoverse_v2': ArgoverseV2DataModule,
    }[args.dataset](**vars(args))
    model_checkpoint = ModelCheckpoint(monitor='val_minFDE', save_top_k=5, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
                         strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True),
                         callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs)
    trainer.fit(model, datamodule)
