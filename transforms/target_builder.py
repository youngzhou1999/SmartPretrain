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
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import subgraph, bipartite_subgraph

from utils import wrap_angle


class TargetBuilder(BaseTransform):

    def __init__(self,
                 num_historical_steps: int,
                 num_future_steps: int) -> None:
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps

    def __call__(self, data: HeteroData) -> HeteroData:
        origin = data['agent']['position'][:, self.num_historical_steps - 1]
        theta = data['agent']['heading'][:, self.num_historical_steps - 1]
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(data['agent']['num_nodes'], 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        data['agent']['target'] = origin.new_zeros(data['agent']['num_nodes'], self.num_future_steps, 4)
        data['agent']['target'][..., :2] = torch.bmm(data['agent']['position'][:, self.num_historical_steps:, :2] -
                                                     origin[:, :2].unsqueeze(1), rot_mat)
        if data['agent']['position'].size(2) == 3:
            data['agent']['target'][..., 2] = (data['agent']['position'][:, self.num_historical_steps:, 2] -
                                               origin[:, 2].unsqueeze(-1))
        data['agent']['target'][..., 3] = wrap_angle(data['agent']['heading'][:, self.num_historical_steps:] -
                                                     theta.unsqueeze(-1))
        return data


############# For ssl ##########################

class TargetBuilder_ssl(BaseTransform):

    def __init__(self,
                 num_historical_steps: int,
                 num_future_steps: int) -> None:
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps

    def __call__(self, data: HeteroData) -> HeteroData:
        last_data = data['t_']
        cur_data = data['t']
        def deal(data):
            pos_start = data['start']+self.num_historical_steps
            origin = data['pos_110'][:, pos_start - 1]
            theta = data['heading_110'][:, pos_start - 1]
            cos, sin = theta.cos(), theta.sin()
            rot_mat = theta.new_zeros(data['agent']['num_nodes'], 2, 2)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = -sin
            rot_mat[:, 1, 0] = sin
            rot_mat[:, 1, 1] = cos
            data['agent']['target'] = origin.new_zeros(data['agent']['num_nodes'], self.num_historical_steps+self.num_future_steps, 4)
            data['agent']['target'][..., :2] = torch.bmm(data['pos_110'][:, :, :2] -
                                                        origin[:, :2].unsqueeze(1), rot_mat)
            if data['pos_110'].size(2) == 3:
                data['agent']['target'][..., 2] = (data['pos_110'][:, :, 2] -
                                                origin[:, 2].unsqueeze(-1))
            data['agent']['target'][..., 3] = wrap_angle(data['heading_110'] -
                                                        theta.unsqueeze(-1))
        deal(last_data)
        deal(cur_data)

        return data


############### can be used to reduce gpu memory ######################
class LaneRandomOcclusion(BaseTransform):
    def __init__(self, lane_occlusion_ratio=0.5):
        super(LaneRandomOcclusion, self).__init__()
        self.lane_occlusion_ratio = lane_occlusion_ratio

    def _mask_edge_index(self, edge_index, occlusion_index):
        mask = ~torch.isin(edge_index[0], occlusion_index)
        return edge_index[:, mask]

    def __call__(self, data):
        last_data = data['t_']
        cur_data = data['t']
        def deal_data(data):

            def gen_sub_set(lo, hi, subset_size, edge_index):
                import random

                # 生成候选集合：从 lower_bound 到 upper_bound 的整数列表
                candidate_set = list(range(lo, hi))

                # 随机选择不重复的子集
                random_subset = random.sample(candidate_set, subset_size)
                random_subset = [i for i in random_subset if i in edge_index]
                random_subset.sort()

                return torch.tensor(random_subset)
            
            if data['data_set'] != 2:
                return

            # polygon at least two
            if data['map_polygon']['num_nodes'] < 2:
                return
            num_exists = int(data['map_polygon']['num_nodes'] * (1-self.lane_occlusion_ratio))
            polygon_subset = gen_sub_set(0, data['map_polygon']['num_nodes'], num_exists, data['map_polygon', 'to', 'map_polygon']['edge_index'])
            
            if len(polygon_subset) > 0:
                edge_index, _, edge_mask = subgraph(polygon_subset, data['map_polygon', 'to', 'map_polygon']['edge_index'], relabel_nodes=True, return_edge_mask=True)
            else:
                # ignore aug
                return
            data['map_polygon', 'to', 'map_polygon']['edge_index'] = edge_index.clone()
            data['map_polygon', 'to', 'map_polygon']['type'] = data['map_polygon', 'to', 'map_polygon']['type'][edge_mask].clone()
            
            polygon_mask = torch.zeros(data['map_polygon']['num_nodes']).bool()
            polygon_mask[polygon_subset] = True

            data['map_polygon']['num_nodes'] = len(polygon_subset)
            data['map_polygon']['position'] = data['map_polygon']['position'][polygon_mask].clone()
            data['map_polygon']['orientation'] = data['map_polygon']['orientation'][polygon_mask].clone()
            data['map_polygon']['type'] = data['map_polygon']['type'][polygon_mask].clone()
            data['map_polygon']['is_intersection'] = data['map_polygon']['is_intersection'][polygon_mask].clone()

            # cut extra points
            pt_2_polygon_mask = torch.isin(data['map_point', 'to', 'map_polygon']['edge_index'][1,:], polygon_subset)
            if pt_2_polygon_mask.sum(-1)==0:
                import pdb
                pdb.set_trace()
            pts_subset = torch.nonzero(pt_2_polygon_mask).squeeze(-1)
            if pts_subset.numel() == 0:
                # 如果 pts_subset 为空，可能需要返回或者执行其他逻辑
                print("No valid points found.")
                return  # 或者执行其他逻辑

            edge_index, _, edge_mask = bipartite_subgraph((pts_subset, polygon_subset), data['map_point', 'to', 'map_polygon']['edge_index'], relabel_nodes=True, return_edge_mask=True)

            data['map_point', 'to', 'map_polygon']['edge_index']=edge_index.clone()

            data['map_point']['num_nodes'] = len(pts_subset)
            data['map_point']['position'] = data['map_point']['position'][pt_2_polygon_mask].clone()
            data['map_point']['orientation'] = data['map_point']['orientation'][pt_2_polygon_mask].clone()
            data['map_point']['type'] = data['map_point']['type'][pt_2_polygon_mask].clone()
            data['map_point']['magnitude'] = data['map_point']['magnitude'][pt_2_polygon_mask].clone()
            data['map_point']['height'] = data['map_point']['height'][pt_2_polygon_mask].clone()
            data['map_point']['side'] = data['map_point']['side'][pt_2_polygon_mask].clone()

        deal_data(cur_data)
        deal_data(last_data)
        return data