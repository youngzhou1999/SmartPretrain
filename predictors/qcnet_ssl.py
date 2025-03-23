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
from itertools import chain
from itertools import compress
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData

from losses import MixtureNLLLoss
from losses import NLLLoss
from metrics import Brier
from metrics import MR
from metrics import minADE
from metrics import minAHE
from metrics import minFDE
from metrics import minFHE
from modules import QCNetDecoder_ssl as QCNetDecoder
from modules import QCNetEncoder

try:
    from av2.datasets.motion_forecasting.eval.submission import ChallengeSubmission
except ImportError:
    ChallengeSubmission = object

import copy
from utils import weight_init
import math

class Branch(nn.Module):
    """
    A network branch that combines an encoder and decoder module,
    followed by a projection head to produce a final embedding.

    Typically used in contrastive/self-supervised learning where
    the projection head maps features to a latent space.
    """

    def __init__(self, enc, dec, input_dim=128):
        """
        Args:
            enc (nn.Module): The encoder module to extract spatial features from input data.
            dec (nn.Module): The decoder module that processes the encoded scene representation.
            input_dim (int): Input feature dimensionality.
        """
        super(Branch, self).__init__()

        self.encoder = enc           # Encoder network (e.g., a GNN or CNN)
        self.decoder = dec           # Decoder network (e.g., another GNN or attention module)
        self.input_dim = input_dim

        hidden_dim = input_dim * 4   # Hidden layer dimension in projection head
        output_dim = input_dim * 2   # Output embedding size

        # Projection head: a 2-layer MLP with BatchNorm and ReLU
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),     # First linear layer
            nn.BatchNorm1d(hidden_dim),                       # Batch normalization
            nn.ReLU(inplace=True),                            # Non-linearity
            nn.Linear(hidden_dim, output_dim, bias=True)      # Final projection
        )

    def forward(self, data):
        """
        Forward pass through encoder, decoder, and projection head.

        Args:
            data (dict or HeteroData): Input scene graph or feature data.

        Returns:
            Tensor: Final projected embedding (batch_size, seq_len, output_dim)
        """

        # Encode the input scene (e.g., graph or spatial structure)
        scene_enc = self.encoder(data)

        # Decode the representation (e.g., aggregate agent information)
        embed = self.decoder(data, scene_enc)

        # Apply first linear layer from projection head
        embed = self.proj[0](embed)

        # Apply BatchNorm1d across the correct dimension
        # Permute to (batch, feature, time) -> BatchNorm -> permute back
        embed = self.proj[1](embed.permute(1, 2, 0)).permute(2, 0, 1)

        # Apply remaining layers (ReLU + Linear)
        embed = self.proj[2:](embed)

        return embed

class QCNet_ssl(pl.LightningModule):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 num_recurrent_steps: int,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_dec_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 submission_dir: str,
                 submission_file_name: str,
                 lr_init: float,
                 lr_last: float,
                 weight_decay_init: float,
                 contra_t:float,
                 contra_momentum: float,
                 warmup: int,
                 **kwargs) -> None:
        super(QCNet_ssl, self).__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps
        self.num_freq_bands = num_freq_bands
        self.num_map_layers = num_map_layers
        self.num_agent_layers = num_agent_layers
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.pl2pl_radius = pl2pl_radius
        self.time_span = time_span
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_t2m_steps = num_t2m_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.submission_dir = submission_dir
        self.submission_file_name = submission_file_name

        encoder = QCNetEncoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_map_layers=num_map_layers,
            num_agent_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        decoder = QCNetDecoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            output_head=output_head,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            num_modes=num_modes,
            num_recurrent_steps=num_recurrent_steps,
            num_t2m_steps=num_t2m_steps,
            pl2m_radius=pl2m_radius,
            a2m_radius=a2m_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_dec_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

        # Initialize the main branch (used for current view) and its momentum version (target network)
        self.branch = Branch(encoder, decoder, input_dim=hidden_dim)
        self.momentum_branch = copy.deepcopy(self.branch)  # Momentum encoder for contrastive learning (target network)

        # ----------------- Contrastive Learning Hyperparameters ------------------ #
        self.T = contra_t                                # Temperature for contrastive loss
        self.contra_momentum = contra_momentum           # Momentum coefficient for encoder update
        self.lr_init = lr_init                            # Initial learning rate
        self.lr_last = lr_last                            # Final learning rate (for cosine decay)
        self.weight_decay_init = weight_decay_init        # Initial weight decay
        self.warmup = warmup                              # Warmup steps for learning rate

        # ----------------- Prediction Heads ------------------ #
        hidden_dim_dec = 1024           # Hidden dimension for projection/prediction MLPs
        input_dim = hidden_dim * 2          # default: projector outputdim is 2*hidden_dim

        # Node-level contrastive predictor
        self.contra_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_dec, bias=True),   # Project concatenated embeddings
            nn.BatchNorm1d(hidden_dim_dec),                         # Normalize batch
            nn.ReLU(inplace=True),                                  # Non-linearity
            nn.Linear(hidden_dim_dec, input_dim, bias=True)     # Predict the other view in embedding space
        )

        # Regression predictor head (e.g., for trajectory prediction or spatial decoding)
        self.rec_predictor = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim_dec, bias=True),    # Input: concatenated features (e.g., pairwise or multiple)
            nn.LayerNorm(hidden_dim_dec),                           # Normalize features
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim_dec, (self.num_historical_steps+self.num_future_steps) * 2, bias=True)            # Predict 110 x 2D outputs (e.g., coordinates)
        )

        # ----------------- Learnable Temporal Embedding ------------------ #
        # Used to provide time-aware information into model (e.g., for decoding sequences)
        self.temp_emb = nn.Parameter(
            torch.zeros((self.num_historical_steps+self.num_future_steps) * 2, input_dim), requires_grad=True     # 110 time steps, each with embedding dim
        )

        self.apply(weight_init)

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.branch.parameters(), self.momentum_branch.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def adjust_moco_momentum(self, epoch):
        """Adjust moco momentum based on current epoch"""
        max_epoch = self.trainer.max_epochs
        m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) * (1. - self.contra_momentum)
        self.log('momentum_m', m, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        return m

    def adjust_learning_rate_decay(self, optimizer, epoch):
        """Decays the learning rate with half-cycle cosine after warmup"""
        warmup_epochs = self.warmup
        max_epochs = self.trainer.max_epochs
        if epoch < warmup_epochs:
            lr = (self.lr-self.lr_init) * epoch / warmup_epochs + self.lr_init
        else:
            lr = self.lr_last + (self.lr-self.lr_last) * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        decay = (self.weight_decay-self.weight_decay_init) * epoch / max_epochs + self.weight_decay_init
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = decay

        self.log('lr', lr, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('decay', decay, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)


    def compute_contrastive_loss(self, embed_t, embed_all, data_mask_t, mask_all):
        """
        Compute contrastive loss between projected anchor embeddings (query)
        and stacked momentum embeddings (keys) across recurrent steps.

        Args:
            embed_t (Tensor): Anchor embeddings (before projection), shape (steps, batch, dim)
            embed_all (Tensor): Stacked momentum embeddings from t and t_, shape (2, steps, batch, dim)
            data_mask_t (BoolTensor): Valid mask for anchor agents, shape (batch,)
            mask_all (BoolTensor): Stacked key masks from t and t_, shape (2, batch)

        Returns:
            contra_loss (Tensor): Averaged contrastive loss across valid recurrent steps.
        """

        # Step 1: Project anchor embeddings through predictor MLP
        contra_embed = self.contra_predictor[0](embed_t)  # First Linear layer
        contra_embed = self.contra_predictor[1](contra_embed.permute(1, 2, 0)).permute(2, 0, 1)  # BatchNorm
        contra_embed = self.contra_predictor[2:](contra_embed)  # ReLU + Linear → shape (steps, batch, dim)

        contra_loss = 0
        contra_count = 0

        # Step 2: Loop through recurrent steps and compute contrastive loss
        for rec_idx in range(self.num_recurrent_steps):
            query_embed = contra_embed[rec_idx]           # (batch, dim)
            source_embed = embed_all[:, rec_idx]          # (2, batch, dim)
            q_mask = data_mask_t                          # (batch,)
            k_mask = mask_all                             # (2, batch)

            # Skip this step if there are no valid positive/negative pairs
            if (q_mask & k_mask[-1]).sum(-1) == 0:
                continue

            # Compute contrastive loss for this step
            contra_loss_i = self.contrastive_loss_2N_minu_1(
                query_embed, source_embed, q_mask, k_mask, prefix=""
            )

            if torch.isnan(contra_loss_i):
                continue

            contra_loss += contra_loss_i
            contra_count += 1

        # Normalize the total loss over number of valid steps
        if contra_count > 0:
            contra_loss /= contra_count

        return contra_loss

    def compute_rec_loss(self, scene_t, scene_t_, embed_t, data_mask_t):
        """
        Compute regression loss between predicted and ground-truth future trajectories
        using a recurrent prediction head over multiple temporal segments.

        Args:
            scene_t (HeteroData): Anchor view scene data (contains 'start', 'target', etc.)
            scene_t_ (HeteroData): Target view scene data (contains 'agent', seg indices, etc.)
            embed_t (Tensor): Embedding from the anchor encoder, shape (num_steps, batch, embed_dim)
            data_mask_t (BoolTensor): Valid agent mask from anchor view, shape (batch,)

        Returns:
            rec_loss (Tensor): Averaged regression loss over valid agents and time steps.
        """

        # Step 1: Get temporal embedding based on start time of anchor view (matched by batch)
        temp_idx = (scene_t['start'])[scene_t_['agent']['batch']].long()
        temp_emb = self.temp_emb[temp_idx].unsqueeze(0).repeat(embed_t.shape[0], 1, 1)  # (steps, batch, dim)

        # Step 2: Predict all steps with reconstruction predictor
        preds = self.rec_predictor(torch.cat((embed_t, temp_emb), dim=-1))  # (steps, batch, 110*2)
        preds = preds.reshape(self.num_recurrent_steps, -1, 110, 2)     # (steps, batch, 110, 2)

        pred_futs = []
        for rec_idx in range(self.num_recurrent_steps):
            pred_i = preds[rec_idx]  # (batch, 110, 2)

            seg_start_2_end = scene_t_.seg_start_2_end

            # Slice segment indices for each recurrence
            if rec_idx == 0:
                seg_start_2_end = seg_start_2_end[:, :10]
            elif rec_idx == 1:
                seg_start_2_end = seg_start_2_end[:, 10:30]
            else:
                seg_start_2_end = seg_start_2_end[:, 30:50]

            # Expand time indices to 2D (for x,y)
            dummy = seg_start_2_end.unsqueeze(2).expand(seg_start_2_end.size(0), seg_start_2_end.size(1), 2)

            # Gather predicted deltas
            pred_fut = torch.gather(pred_i, 1, dummy)  # (batch, time, 2)
            pred_futs.append(pred_fut)

        # Step 3: Combine predicted segments and convert to absolute position using cumsum
        pred_fut = torch.cat(pred_futs, dim=1)               # (batch, 50, 2)
        pred_fut = torch.cumsum(pred_fut, dim=-2)            # convert deltas to coordinates

        # Step 4: Gather ground-truth target positions
        seg_start_2_end = scene_t_.seg_start_2_end
        dummy = seg_start_2_end.unsqueeze(2).expand(seg_start_2_end.size(0), seg_start_2_end.size(1), 2)
        gt = scene_t['agent']['target'][..., :2]
        gt = torch.gather(gt, 1, dummy)                      # (batch, 50, 2)

        # Step 5: Apply prediction mask from target view & data mask from anchor view
        pred_mask = scene_t_['agent']['predict_mask_ssl']    # (batch, 50)
        pred_mask = data_mask_t.unsqueeze(-1) & pred_mask    # (batch, 50)

        # Step 6: Compute Smooth L1 loss (Huber loss)
        rec_loss = nn.SmoothL1Loss(reduction='none')(pred_fut, gt).sum(-1)  # (batch, 50)
        rec_loss = rec_loss * pred_mask                                     # apply mask

        # Step 7: Normalize loss over time per agent, then average across agents
        rec_loss = rec_loss.sum(dim=0) / pred_mask.sum(dim=0).clamp(min=1)
        rec_loss = rec_loss.mean()

        return rec_loss

    def forward(self, data: HeteroData, prefix=''):

        # Split input data into two views (target & anchor)
        scene_t_ = data['t_']  # View t_ (usually later in time)
        scene_t = data['t']    # View t  (anchor view)

        # Compute momentum encoder embedding for target view (t_) without gradient
        with torch.no_grad():
            embed_t_ = self.momentum_branch(data=scene_t_)

        # Get binary validity mask for target view (which agents are valid)
        data_mask_t_ = scene_t_.data_mask

        # Compute current encoder embedding for anchor view (t)
        embed_t = self.branch(data=scene_t)

        # Get binary validity mask for anchor view
        data_mask_t = scene_t.data_mask

        # Optionally get momentum embedding of anchor view itself (used for 2N setup)
        with torch.no_grad():
            momentum_embed_t = self.momentum_branch(data=scene_t)

        # Get dataset info for each agent instance and repeat across views
        data_set = scene_t['data_set']
        data_set = data_set[scene_t['agent']['batch']]
        data_set = data_set.repeat(2)

        # Combine both momentum embeddings (t and t_) for contrastive "keys"
        embed_all = torch.cat(
            (momentum_embed_t.unsqueeze(0), embed_t_.unsqueeze(0)), dim=0
        )  # Shape: (2, num_steps, batch_size, embed_dim)

        # Combine corresponding validity masks
        mask_all = torch.cat(
            (data_mask_t.unsqueeze(0), data_mask_t_.unsqueeze(0)), dim=0
        )  # Shape: (2, batch_size)

        contra_loss = self.compute_contrastive_loss(embed_t, embed_all, data_mask_t, mask_all)
        
        rec_loss = self.compute_rec_loss(scene_t, scene_t_, embed_t, data_mask_t)

        if prefix == '':
            if contra_loss == 0:
                import pdb
                pdb.set_trace()
            self.log('component/contra_loss', contra_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('component/rec_loss', rec_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            return contra_loss+rec_loss
        else:
            self.log(prefix+'component/contra_loss', contra_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log(prefix+'component/rec_loss', rec_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            return contra_loss+rec_loss

    def training_step(self,
                      data,
                      batch_idx):
        epoch = self.current_epoch
        max_iteration = len(self.trainer.train_dataloader)
        current_iteration = batch_idx
        if self.warmup:
            assert self.warmup != 0
            self.adjust_learning_rate_decay(self.optimizers(), int(epoch)+current_iteration/max_iteration)
        epoch = self.current_epoch
        max_iteration = len(self.trainer.train_dataloader)
        m = self.adjust_moco_momentum(int(epoch)+current_iteration/max_iteration)
        # if current_iteration % self.contra_freq == 0:
        self._update_momentum_encoder(m)  # update the momentum encoder

        res = self(data)
        self.log('loss', res, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        return res

    def validation_step(self,
                        data,
                        batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        
        res = self(data, 'val_')
        self.log('val_loss', res, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        return res

    def contrastive_loss_2N_minu_1(self, q, k, q_mask, k_mask, prefix='', data_set=None):
        """
        Compute 2N contrastive loss (InfoNCE) with optional dataset-aware loss filtering,
        supporting distributed training with synchronized negatives.

        Args:
            q (Tensor): Query embeddings, shape (batch, dim)
            k (Tensor): Key embeddings from momentum encoder, shape (2, batch, dim)
            q_mask (BoolTensor): Validity mask for queries, shape (batch,)
            k_mask (BoolTensor): Validity mask for keys, shape (2, batch)
            prefix (str): Optional logging prefix
            data_set (Tensor, optional): Dataset indicator per sample, used for loss partitioning

        Returns:
            contrastive_loss (Tensor): Averaged contrastive loss over valid samples
        """

        # Use only the second key view's mask (typically t_ view) for filtering
        tar_mask = k_mask[-1].clone()

        # Reshape keys and masks from (2, batch, dim) → (2*batch, dim)
        bz_q = q.shape[0]
        k = k.reshape(2 * bz_q, -1)
        k_mask = k_mask.reshape(2 * bz_q)

        # Normalize query and key embeddings
        q = nn.functional.normalize(q, dim=-1)
        k = nn.functional.normalize(k, dim=-1)

        # --------- Distributed All-Gather Setup --------- #
        # Get the number of samples across all ranks
        bz = k.shape[0]
        bz = self.all_gather(bz)  # shape: (world_size,)
        bz_max = bz.max()

        # Build a validity mask tensor for padding consistency
        valid_mask = []
        rank_idx = []
        idx_i = 0
        assert bz[torch.distributed.get_rank()] == k.shape[0]

        for bz_i in bz:
            v_mask = k.new_zeros(bz_max.item(), dtype=torch.bool)
            v_mask[:bz_i.item()] = True
            valid_mask.append(v_mask)
            rank_idx.append(idx_i)
            idx_i += bz_i.item()

        valid_mask_tensor = torch.cat(valid_mask, dim=-1)  # shape: (total padded keys,)

        # Pad and gather all key embeddings across GPUs
        k = torch.cat((k, torch.zeros(bz_max - k.shape[0], k.shape[1]).to(k.device)), dim=0)
        k = self.all_gather(k, sync_grads=True)  # shape: (world_size, bz_max, dim)
        num_all = k.shape[0] * k.shape[1]
        k = k.reshape(num_all, -1)
        k = k[valid_mask_tensor]  # keep only valid rows

        # Same gathering and filtering for k_mask
        k_mask = torch.cat((k_mask, torch.zeros(bz_max - k_mask.shape[0]).bool().to(k.device)), dim=0)
        k_mask = self.all_gather(k_mask, sync_grads=True)  # shape: (world_size, bz_max)
        k_mask = k_mask.reshape(num_all)
        k_mask = k_mask[valid_mask_tensor]

        valid_num_all = k_mask.sum(-1)

        # Logging total number of valid negatives
        self.log(prefix + 'num_neg_sample', valid_num_all.float(), prog_bar=False,
                on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        # --------- Compute similarity scores (logits) --------- #
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T  # shape: (batch, all_keys)

        # Handle distributed self-positive masking
        N = logits.shape[0]  # number of queries in local batch
        self_mask = logits.new_zeros(logits.shape).bool()
        self_idx = (torch.arange(N, dtype=torch.long) + rank_idx[torch.distributed.get_rank()]).to(q.device)
        self_mask[torch.arange(N), self_idx] = True
        logits.masked_fill_(self_mask, -float('inf'))  # prevent self-matching

        # Mask out invalid negatives using gathered k_mask
        invalid_mask = (~k_mask).repeat(N).reshape(N, -1).bool()
        logits.masked_fill_(invalid_mask, -float('inf'))

        # Compute matching labels: match i-th query to i-th positive (offset by rank start index)
        rank_prefix = valid_mask[torch.distributed.get_rank()].sum(-1).item() // 2 + rank_idx[torch.distributed.get_rank()]
        labels = torch.arange(N, dtype=torch.long).to(q.device) + rank_prefix

        # Filter out invalid query-key pairs where either side is invalid
        mask_two = q_mask & tar_mask  # shape: (batch,)

        # --------- Optional: Per-dataset contrastive loss --------- #
        if data_set is not None:
            contra_loss = 0
            count = 0

            # Gather and align dataset info for all keys
            data_set_all = torch.cat(
                (data_set, torch.zeros(bz_max - data_set.shape[0]).long().to(data_set.device)), dim=0
            )
            data_set_all = self.all_gather(data_set_all, sync_grads=True)
            data_set_all = data_set_all.reshape(-1)[valid_mask_tensor]  # shape: (all valid keys,)

            for i in range(3):  # Assume 3 datasets: 0, 1, 2
                loss_mask = data_set == i                  # mask for current dataset (queries)
                data_set_mask = (data_set_all == i).repeat(N).reshape(N, -1)  # keys

                logits_i = logits.clone()
                logits_i.masked_fill_(~data_set_mask, -float('inf'))  # filter keys not from dataset i

                mask_all = mask_two & loss_mask[:N]  # filter valid queries from dataset i
                logits_i = logits_i[mask_all]
                labels_i = labels[mask_all]

                if i == 1:
                    acc = (F.softmax(logits_i, dim=-1).argmax(-1) == labels_i).sum() / logits_i.shape[0]
                    self.log(prefix + 'contra_acc', acc, prog_bar=False, on_step=False,
                            on_epoch=True, batch_size=logits_i.size(0), sync_dist=True)

                if logits_i.shape[0] == 0:
                    continue

                count += 1
                contra_loss += nn.CrossEntropyLoss()(logits_i, labels_i) * (2 * self.T)

            return contra_loss / count if count > 0 else torch.tensor(0.0).to(q.device)

        else:
            # Standard (non-dataset-specific) contrastive loss
            logits = logits[mask_two]
            labels = labels[mask_two]

            acc = (F.softmax(logits, dim=-1).argmax(-1) == labels).sum() / logits.shape[0]
            self.log(prefix + 'contra_acc', acc, prog_bar=False, on_step=False,
                    on_epoch=True, batch_size=logits.size(0), sync_dist=True)

            return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    
    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.SyncBatchNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        for param_b, param_m in zip(self.branch.parameters(), self.momentum_branch.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('QCNet')
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--input_dim', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--output_dim', type=int, default=2)
        parser.add_argument('--output_head', action='store_true')
        parser.add_argument('--num_historical_steps', type=int, required=True)
        parser.add_argument('--num_future_steps', type=int, required=True)
        # mode set to 1 for pretrain
        parser.add_argument('--num_modes', type=int, default=1)
        parser.add_argument('--num_recurrent_steps', type=int, required=True)
        parser.add_argument('--num_freq_bands', type=int, default=64)
        parser.add_argument('--num_map_layers', type=int, default=1)
        parser.add_argument('--num_agent_layers', type=int, default=2)
        parser.add_argument('--num_dec_layers', type=int, default=2)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--head_dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--pl2pl_radius', type=float, required=True)
        parser.add_argument('--time_span', type=int, default=None)
        parser.add_argument('--pl2a_radius', type=float, required=True)
        parser.add_argument('--a2a_radius', type=float, required=True)
        parser.add_argument('--num_t2m_steps', type=int, default=None)
        parser.add_argument('--pl2m_radius', type=float, required=True)
        parser.add_argument('--a2m_radius', type=float, required=True)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        parser.add_argument('--submission_dir', type=str, default='./')
        parser.add_argument('--submission_file_name', type=str, default='submission')

        parser.add_argument('--lr_init', type=float, default=1e-4)
        parser.add_argument('--lr_last', type=float, default=1e-6)
        parser.add_argument('--weight_decay_init', type=float, default=1e-4)
        parser.add_argument('--contra_t', type=float, default=0.07)
        parser.add_argument('--contra_momentum', type=float, default=0.996)
        parser.add_argument('--warmup', type=int, default=15)
        return parent_parser
