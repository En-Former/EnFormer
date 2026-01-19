import copy
from typing import Tuple, Optional, Type, Sequence, Union, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.nn.common_types import _size_2_t, _ratio_2_t
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from timm.models import register_model
from timm.layers import DropPath, trunc_normal_, to_2tuple

from .utils import *
from .base_clustering import *

try:
    import math
    from collections import OrderedDict
    from mmcv.runner import _load_checkpoint
    from mmcv.cnn import constant_init, trunc_normal_init, normal_init
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger as det_get_root_logger
    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first.")
    has_mmdet = False

try:
    if not has_mmdet:
        import math
        from collections import OrderedDict
        from mmcv.runner import _load_checkpoint
        from mmcv.cnn import constant_init, trunc_normal_init, normal_init
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger as seg_get_root_logger
    has_mmseg = True
except ImportError:
    print("If for semantic segmentation, please install mmsegmentation first.")
    has_mmseg = False


DEFAULT_CFG = {
    'input_size': (3, 224, 224), 'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
    'crop_pct': DEFAULT_CROP_PCT, 'interpolation': 'bicubic',
}


class Stem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t = 3):
        super().__init__()
        self.aggregate1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size, 2, kernel_size // 2)
        self.norm1 = LayerNorm(out_channels // 2, data_format='channels_first')
        self.act = nn.SiLU(inplace=True)
        self.aggregate2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size, 2, kernel_size // 2)
        self.norm2 = LayerNorm(out_channels, data_format='channels_first')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm2(self.aggregate2(self.act(self.norm1(self.aggregate1(x)))))


class PointAggregation(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: _size_2_t = 3,
                 stride: _size_2_t = 2, padding: _size_2_t = 0):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.aggregate = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = LayerNorm(embed_dim, data_format='channels_first')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aggregate(x)
        x = self.norm(x)
        return x


class EnsembleClustering(nn.Module):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, num_heads: int = 4, head_dim: int = 24,
                 use_agents: bool = True, agents_hw: Optional[Tuple[int, int]] = (7, 7),
                 clusters_hw: Union[Tuple[int, int], List[Tuple[int, int]]] = (2, 2),
                 cluster_modules: List[str] = ['PartitionalClustering', 'FuzzyClustering'],
                 with_cluster_bias: bool = True, with_cluster_position: bool = False, use_cosine: bool = False,
                 cluster_cfgs: Optional[List[dict]] = None, layer_scale: Optional[float] = None, deploy: bool = False):
        super().__init__()
        out_channels = out_channels or in_channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_agents = use_agents
        self.deploy = deploy
        self.with_cluster_bias = with_cluster_bias
        self.with_cluster_position = with_cluster_position
        self.use_cosine = use_cosine
        if self.use_agents:
            self.agents = nn.AdaptiveAvgPool2d(agents_hw)
            self.agents_h, self.agents_w = agents_hw
            self.num_agents = agents_hw[0] * agents_hw[1]

        self.num_clustering_modules = len(cluster_modules)
        if isinstance(clusters_hw, list):
            assert len(clusters_hw) == self.num_clustering_modules
        else:
            clusters_hw = [clusters_hw] * self.num_clustering_modules
        if isinstance(cluster_cfgs, list):
            if len(cluster_cfgs) == 1:
                cluster_cfgs = [cluster_cfgs[0]] * self.num_clustering_modules
            else:
                assert len(cluster_cfgs) == self.num_clustering_modules
        else:
            cluster_cfgs = [{}] * self.num_clustering_modules
        num_all_clusters = sum([hw[0] * hw[1] for hw in clusters_hw])

        self.proj = Linear(in_channels, (1 + 2 * len(cluster_modules)) * self.num_heads * self.head_dim, True, False, None,
                            deploy, True, True)

        for i in range(self.num_clustering_modules):
            module = cluster_modules[i]
            setattr(self, f'num_clusters_{i}', clusters_hw[i][0] * clusters_hw[i][1])
            setattr(self, f'cluster_module_{i}',
                    build_clustering(module, dim=self.head_dim, num_clusters=getattr(self, f'num_clusters_{i}'),
                                     return_clusters=True, with_cluster_bias=self.with_cluster_bias,
                                     with_cluster_position=self.with_cluster_position, use_cosine=self.use_cosine,
                                     **cluster_cfgs[i]))
            setattr(self, f'clusters_{i}', nn.AdaptiveAvgPool2d(clusters_hw[i]))

            if self.with_cluster_position and self.with_cluster_bias:
                h, w = clusters_hw[i]
                ref_y = torch.arange(-0.5, h - 0.5) / (h - 1.0)
                ref_x = torch.arange(-0.5, w - 0.5) / (w - 1.0)
                ref_grid_y, ref_grid_x = torch.meshgrid(ref_y, ref_x, indexing='ij')
                ref_grid = torch.stack((ref_grid_x, ref_grid_y), dim=-1).view(1, h * w, 2)
                self.register_buffer(f'clusters_position_{i}', ref_grid)

                offset_scale = torch.tensor([0.5 / w, 0.5 / h])
                self.register_buffer(f'offset_scale_{i}', offset_scale)
                setattr(self, f'offset_param_{i}', nn.Parameter(torch.zeros(1, h * w, 2)))

        self.sim_alpha = nn.Parameter(torch.ones(num_all_clusters))
        self.sim_beta = nn.Parameter(torch.zeros(num_all_clusters))

        self.proj2 = Linear(self.num_heads * self.head_dim, out_channels, True, False, layer_scale, deploy, True, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        n = 1 + 2 * self.num_clustering_modules
        d = self.num_heads * self.head_dim
        x = self.proj(x).view(B, n, d, H, W).permute(1, 0, 2, 3, 4)

        if self.num_heads > 1:
            x = x.reshape(n, B * self.num_heads, self.head_dim, H, W)
        p, r = x[0], x[1:]  # [B * num_heads, head_dim, H, W], [n - 1, B * num_heads, head_dim, H, W]
        _, b, c, _, _ = r.shape
        r_group = r.view(self.num_clustering_modules, 2, b, c, H, W)  # [self.num_clustering_modules, 2, B * num_heads, head_dim, H, W]

        agg_list = []
        for i in range(self.num_clustering_modules):
            k, v = r_group[i]  # [b, c, H, W], [b, c, H, W]
            k_clusters = getattr(self, f'clusters_{i}')(k).view(b, c, -1).permute(0, 2, 1)  # [b, M, c]
            v_clusters = getattr(self, f'clusters_{i}')(v).view(b, c, -1).permute(0, 2, 1)  # [b, M, c]

            if self.with_cluster_position:
                bounded_offset = torch.tanh(getattr(self, f'offset_param_{i}')) * getattr(self, f'offset_scale_{i}')
                clusters_position = getattr(self, f'clusters_position_{i}') + bounded_offset  # [1, M, 2]
                k_clusters = torch.cat((k_clusters, clusters_position.expand(b, -1, -1)), dim=-1)  # [b, M, c+2]
                v_clusters = torch.cat((v_clusters, clusters_position.expand(b, -1, -1)), dim=-1)  # [b, M, c+2]

            if self.use_agents:
                k = self.agents(k).view(b, c, -1)
                v = self.agents(v).view(b, c, -1).permute(0, 2, 1)
            else:
                k = k.view(b, c, -1)
                v = v.view(b, c, -1).permute(0, 2, 1)
            agg = getattr(self, f'cluster_module_{i}')(k, k_clusters, v, v_clusters)  # [b, M, c]
            agg_list.append(agg)
        agg = torch.cat(agg_list, dim=1)  # [b, M * num_modules, c]

        if self.use_agents:
            p = self.agents(p).view(b, c, -1)  # [b, c, num_agents], set N = num_agents
            h, w = self.agents_h, self.agents_w
        else:
            p = p.view(b, c, -1)  # [b, c, N]
            h, w = H, W

        similarity = self.sim_alpha.view(1, -1, 1) * compute_similarity(agg, p) + self.sim_beta.view(1, -1, 1)  # [b, M * num_modules, N]
        assignment = F.softmax(similarity, dim=1)  # [b, M * num_modules, N]
        if self.training:
            x = (agg.permute(0, 2, 1).unsqueeze(-1) * assignment.unsqueeze(1)).sum(dim=2)
        else:
            x = (agg.permute(0, 2, 1) @ assignment)
        x = x.view(b, c, h, w)
        if self.use_agents:
            x = F.interpolate(x, size=(H, W), mode='bilinear')
        if self.num_heads > 1:
            x = x.view(B, -1, H, W)
        x = self.proj2(x)
        return x

    @torch.no_grad()
    def reparameterize(self):
        for i in range(self.num_clustering_modules):
            cluster_module = getattr(self, f'cluster_module_{i}')
            if hasattr(cluster_module, 'reparameterize'):
                cluster_module.reparameterize()
        self.proj.reparameterize()
        self.proj2.reparameterize()
        self.deploy = True


class ClusterBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, num_heads: int = 4, head_dim: int = 24,
                 use_agents: bool = True, agents_hw: Optional[Tuple[int, int]] = (7, 7),
                 clusters_hw: Union[Tuple[int, int], List[Tuple[int, int]]] = (2, 2),
                 cluster_modules: List[str] = ['ParitionalClustering', 'FuzzyClustering'],
                 with_cluster_bias: bool = True, with_cluster_position: bool = False, use_cosine: bool = False,
                 cluster_cfgs: Optional[List[dict]] = None, ffn_ratio=4.,
                 ffn_first_dw_kernel_size: Optional[_size_2_t] = None, ffn_hidden_dw_kernel_size: Optional[_size_2_t] = None,
                 ffn_act_layer: Type[nn.Module] = nn.SiLU, ffn_drop_rate: _ratio_2_t = 0.,
                 drop_path_rate: _ratio_2_t = 0., layer_scale: float = 1e-6, deploy: bool = False,
                 add_identity: bool = True, with_cp: bool = False):
        super().__init__()
        out_channels = out_channels or in_channels
        drop_path_rate = to_2tuple(drop_path_rate)

        self.cluster = EnsembleClustering(in_channels, in_channels, num_heads, head_dim,
                                          use_agents, agents_hw, clusters_hw, cluster_modules,
                                          with_cluster_bias, with_cluster_position, use_cosine,
                                          cluster_cfgs, layer_scale, deploy)
        self.ffn = FFNx(in_channels, out_channels, ffn_ratio, ffn_first_dw_kernel_size,
                       ffn_hidden_dw_kernel_size, ffn_act_layer, ffn_drop_rate, layer_scale, deploy)
        self.norm1 = LayerNorm(in_channels, data_format="channels_first")
        self.norm2 = LayerNorm(in_channels, data_format="channels_first")
        self.drop_path1 = DropPath(drop_path_rate[0]) if drop_path_rate[0] > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate[1]) if drop_path_rate[1] > 0 else nn.Identity()

        self.add_identity = add_identity and in_channels == out_channels
        self.with_cp = with_cp
        self.deploy = deploy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def _forward(_x):
            if self.add_identity:
                _x = _x + self.drop_path1(self.cluster(self.norm1(_x)))
                _x = _x + self.drop_path2(self.ffn(self.norm2(_x)))
            else:
                _x = self.drop_path1(self.cluster(self.norm1(_x)))
                _x = self.drop_path2(self.ffn(self.norm2(_x)))
            return _x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_forward, x)
        else:
            x = _forward(x)
        return x

    @torch.no_grad()
    def reparameterize(self):
        self.cluster.reparameterize()
        self.ffn.reparameterize()
        self.deploy = True


class ClusterStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, num_heads: int = 4, head_dim: int = 24,
                 use_agents: bool = True, agents_hw: Optional[Tuple[int, int]] = (7, 7),
                 clusters_hw: Union[Tuple[int, int], List[Tuple[int, int]]] = (2, 2),
                 cluster_modules: List[str] = ['ParitionalClustering', 'FuzzyClustering'],
                 with_cluster_bias: bool = True, with_cluster_position: bool = False, use_cosine: bool = False,
                 cluster_cfgs: Optional[List[dict]] = None, ffn_ratio=4.,
                 ffn_first_dw_kernel_size: Optional[_size_2_t] = None, ffn_hidden_dw_kernel_size: Optional[_size_2_t] = None,
                 ffn_act_layer: Type[nn.Module] = nn.SiLU, ffn_drop_rate: _ratio_2_t = 0.,
                 drop_path_rate: Union[Sequence[float], Sequence[Sequence[float]]] = 0.,
                 layer_scale: float = 1e-6, deploy: bool = False, add_identity: bool = True,
                 with_cp: bool = False, downsample: bool = True):
        super().__init__()
        self.downsample = downsample
        self.deploy = deploy
        assert num_blocks >= 1
        if isinstance(ffn_first_dw_kernel_size, int):
            ffn_first_dw_kernel_size = [ffn_first_dw_kernel_size, ] * num_blocks
        elif ffn_first_dw_kernel_size is None:
            ffn_first_dw_kernel_size = [None, ] * num_blocks
        else:
            assert len(ffn_first_dw_kernel_size) == num_blocks
        if isinstance(ffn_hidden_dw_kernel_size, int):
            ffn_hidden_dw_kernel_size = [ffn_hidden_dw_kernel_size, ] * num_blocks
        elif ffn_hidden_dw_kernel_size is None:
            ffn_hidden_dw_kernel_size = [None, ] * num_blocks
        else:
            assert len(ffn_hidden_dw_kernel_size) == num_blocks

        self.blocks = nn.ModuleList([])
        for i in range(num_blocks):
            block = ClusterBlock(in_channels, in_channels, num_heads, head_dim, use_agents, agents_hw, clusters_hw,
                                 cluster_modules, with_cluster_bias, with_cluster_position, use_cosine, cluster_cfgs,
                                 ffn_ratio, ffn_first_dw_kernel_size[i], ffn_hidden_dw_kernel_size[i], ffn_act_layer,
                                 ffn_drop_rate, drop_path_rate[i], layer_scale, deploy, add_identity, with_cp)
            self.blocks.append(block)

        if self.downsample:
            self.down = PointAggregation(in_channels, out_channels, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x)
        if self.downsample:
            x_down = self.down(x)
            return x_down, x
        else:
            return x, x

    @torch.no_grad()
    def reparameterize(self):
        for block in self.blocks:
            block.reparameterize()
        self.deploy = True


class EnsembleFormer(nn.Module):
    classification_arch_settings = {
        # from left to right: (indices)
        # 'arch': [[dim(0), num_blocks(1), num_heads(2), head_dim(3), use_agents(4), agents_hw(5), clusters_hw(6), cluster_modules(7), with_cluster_bias(8), with_cluster_position(9), use_cosine(10),
        #           cluster_cfgs(11), ffn_ratio(12), ffn_first_dw_kernel_size(13), ffn_hidden_dw_kernel_size(14), ffn_drop_rate(15), layer_scale(16), add_identity(17), downsample(18)], ...]
        'S': [[40, 2, 1, 20, True, (14, 14), [(6, 6), (5, 5)], ['FuzzyClustering', 'PartitionalClustering'], False, False, True,
               None, 4., 3, None, (0., 0.), 1e-6, True, True],
              [80, 2, 2, 20, True, (14, 14), [(5, 5), (4, 4)], ['FuzzyClustering', 'PartitionalClustering'], False, False, True,
               None, 4., 3, None, (0., 0.), 1e-6, True, True],
              [160, 10, 4, 20, False, None, [(4, 4), (3, 3)], ['FuzzyClustering', 'PartitionalClustering'], False, False, True,
               None, 4., 3, None, (0., 0.), 1e-6, True, True],
              [320, 4, 8, 20, False, None, [(3, 3), (2, 2)], ['FuzzyClustering', 'PartitionalClustering'], False, False, True,
               None, 4., 3, None, (0., 0.), 1e-6, True, False]],

        'B': [[64, 2, 1, 32, True, (14, 14), [(6, 6), (5, 5)], ['FuzzyClustering', 'PartitionalClustering'], False, False, True,
               None, 4., 3, None, (0., 0.), 1e-6, True, True],
              [128, 2, 2, 32, True, (14, 14), [(5, 5), (4, 4)], ['FuzzyClustering', 'PartitionalClustering'], False, False, True,
               None, 4., 3, None, (0., 0.), 1e-6, True, True],
              [320, 6, 5, 32, False, None, [(4, 4), (3, 3)], ['FuzzyClustering', 'PartitionalClustering'], False, False, True,
               None, 4., 5, None, (0., 0.), 1e-6, True, True],
              [512, 2, 8, 32, False, None, [(3, 3), (2, 2)], ['FuzzyClustering', 'PartitionalClustering'], False, False, True,
               None, 4., 5, None, (0., 0.), 1e-6, True, False]],

        'L': [[64, 2, 1, 32, True, (14, 14), [(6, 6), (5, 5)], ['FuzzyClustering', 'PartitionalClustering'], False, False, True,
               None, 4., 3, None, (0., 0.), 1e-6, True, True],
              [128, 3, 2, 32, True, (14, 14), [(5, 5), (4, 4)], ['FuzzyClustering', 'PartitionalClustering'], False, False, True,
               None, 4., 5, None, (0., 0.), 1e-6, True, True],
              [320, 14, 5, 32, False, None, [(4, 4), (3, 3)], ['FuzzyClustering', 'PartitionalClustering'], False, False, True,
               None, 4., 7, None, (0., 0.), 1e-6, True, True],
              [512, 4, 8, 32, False, None, [(3, 3), (2, 2)], ['FuzzyClustering', 'PartitionalClustering'], False, False, True,
               None, 4., 7, None, (0., 0.), 1e-6, True, False]],
    }

    def __init__(self, in_channels: int = 3, num_classes: int = 1000, arch: str = 'S',
                 arch_setting: Optional[Sequence[list]] = None, task: str = 'classification',
                 out_indices=(3,), drop_path_rate: _ratio_2_t = (0., 0.), with_positional_encoding: bool = True,
                 act_layer: Type[nn.Module] = nn.SiLU, with_cp: bool = False, frozen_stages: int = -1,
                 norm_eval: bool = False, deploy: bool = False, init_cfg=None, pretrained=None, **kwargs):
        super().__init__()

        self.task = task
        self.norm_eval = norm_eval
        self.with_positional_encoding = with_positional_encoding

        if self.task == 'classification':
            assert len(out_indices) == 1, 'For task classification, out_indices must be a single value.'
            arch_setting = arch_setting or self.classification_arch_settings[arch]
        elif self.task == 'detection':
            arch_setting = arch_setting or self.detection_arch_settings[arch]
        elif self.task == 'segmentation':
            arch_setting = arch_setting or self.segmentation_arch_settings[arch]
        else:
            raise NotImplementedError(f'Task {self.task} is not supported.')
        assert set(out_indices).issubset(i for i in range(len(arch_setting)))
        self.out_indices = out_indices

        if self.with_positional_encoding:
            self.in_channels = in_channels + 2
        else:
            self.in_channels = in_channels

        if frozen_stages not in range(-1, len(arch_setting)):
            raise ValueError(f'frozen_stages must be in range(-1, len(arch_setting)). But received {frozen_stages}')
        self.frozen_stages = frozen_stages
        self.deploy = deploy

        self.stem = Stem(self.in_channels, arch_setting[0][0], kernel_size=3)

        depths = [x[1] for x in arch_setting]
        drop_path_rate = to_2tuple(drop_path_rate)
        dpr1 = [x.item() for x in torch.linspace(0, drop_path_rate[0], sum(depths))]
        dpr2 = [x.item() for x in torch.linspace(0, drop_path_rate[1], sum(depths))]
        dpr = [(float(dpr1[i]), float(dpr2[i])) for i in range(sum(depths))]

        self.stages = nn.ModuleList()
        num_stages = len(arch_setting)
        for i, (dim, num_blocks, num_heads, head_dim, use_agents, agents_hw, clusters_hw, cluster_modules,
                with_cluster_bias, with_cluster_position, use_cosine, cluster_cfgs,
                ffn_ratio, ffn_first_dw_kernel_size, ffn_hidden_dw_kernel_size, ffn_drop_rate, layer_scale,
                add_identity, downsample) in enumerate(arch_setting):
            if i == num_stages - 1:
                out_dim = dim
            else:
                out_dim = arch_setting[i + 1][0]
            stage = ClusterStage(dim, out_dim, num_blocks, num_heads, head_dim, use_agents, agents_hw, clusters_hw,
                                 cluster_modules, with_cluster_bias, with_cluster_position, use_cosine,
                                 cluster_cfgs, ffn_ratio, ffn_first_dw_kernel_size, ffn_hidden_dw_kernel_size,
                                 act_layer, ffn_drop_rate, dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                 layer_scale, deploy, add_identity, with_cp, downsample)
            self.stages.append(stage)

            if self.task != 'classification' and i in self.out_indices:
                norm = LayerNorm(dim, data_format="channels_first")
                setattr(self, f'post_norm{i}', norm)

        if self.task == 'classification':
            self.num_classes = num_classes

            last_channels = arch_setting[self.out_indices[-1]][0]
            self.classifier_pre_norm = LayerNorm(last_channels, data_format="channels_last")
            self.head = nn.Linear(last_channels, self.num_classes)
            self.apply(self.pretrain_init_weights)
        else:
            self.init_cfg = copy.deepcopy(init_cfg)
            if self.init_cfg is not None or pretrained is not None:
                self.init_weights()

    def forward(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        return x

    def forward_embeddings(self, x):
        b, _, h, w = x.shape
        if self.with_positional_encoding:
            y_coord = torch.arange(-0.5, h - 0.5) / (h - 1.0)
            x_coord = torch.arange(-0.5, w - 0.5) / (w - 1.0)
            grid = torch.stack(torch.meshgrid(y_coord, x_coord, indexing='ij'), dim=-1)
            x = torch.cat([x, grid.to(x).permute(2, 0, 1).unsqueeze(0).expand(b, -1, h, w)], dim=1)
        x = self.stem(x)
        return x

    def forward_tokens(self, x):
        if self.task == 'classification':
            for stage in self.stages:
                x, _ = stage(x)
            x = self.classifier_pre_norm(x.mean([-2, -1]))
            x = self.head(x)
            return x
        else:
            outs = []
            for i, stage in enumerate(self.stages):
                x, out = stage(x)
                if i in self.out_indices:
                    norm = getattr(self, f'post_norm{i}')
                    outs.append(norm(out))
            return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()

    @staticmethod
    def pretrain_init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, mean=0., std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm, nn.GroupNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        if self.task == 'detection':
            logger = det_get_root_logger()
        elif self.task == 'segmentation':
            logger = seg_get_root_logger()
        else:
            from mmcv.utils import get_logger
            logger = get_logger(name='logger')

        if self.init_cfg is None and pretrained is None:
            logger.warning(f'No pre-trained weights for {self.__class__.__name__}, training start from scratch.')
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, (LayerNorm, nn.LayerNorm, nn.GroupNorm)):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            assert 'checkpoint' in self.init_cfg, (f'Only support specify `Pretrained` in init_cfg` '
                                                   f'in {self.__class__.__name__}')

            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(ckpt_path, logger=logger, map_location='cpu')
            init_state_dict = self.state_dict()
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k in init_state_dict:
                    if v.shape == init_state_dict[k].shape:
                        state_dict[k] = v
                    else:
                        if any(name in k for name in ['sim_alpha', 'sim_beta']):
                            src_tensor = v
                            tgt_tensor = init_state_dict[k].clone()
                            src_flatten = src_tensor.flatten()
                            tgt_flatten = tgt_tensor.flatten()
                            src_mean = src_flatten.mean()
                            tgt_len = tgt_flatten.numel()
                            src_len = src_flatten.numel()

                            if src_len >= tgt_len:
                                tgt_flatten.copy_(src_flatten[:tgt_len])
                            else:
                                tgt_flatten[:src_len].copy_(src_flatten)
                                tgt_flatten[src_len:].fill_(src_mean)
                            state_dict[k] = tgt_flatten.reshape_as(tgt_tensor)
                        elif any(name in k for name in ['clusters_bias']):
                            src_tensor = v
                            tgt_tensor = init_state_dict[k].clone()
                            src_mean = src_tensor.mean().item()

                            if src_tensor.size(0) >= tgt_tensor.size(0):
                                tgt_tensor.copy_(src_tensor[:tgt_tensor.size(0), ...])
                            else:
                                tgt_tensor[:src_tensor.size(0), ...].copy_(src_tensor)
                                tgt_tensor[src_tensor.size(0):, ...].fill_(src_mean)
                            state_dict[k] = tgt_tensor

            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            logger.warning(f'missing_keys: {missing_keys}')
            logger.warning(f'unexpected_keys: {unexpected_keys}')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                m = self.stages[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    @torch.no_grad()
    def reparameterize(self):
        for stage in self.stages:
            stage.reparameterize()
        self.deploy = True

    def _test_equivalency(self, x):
        self.eval()
        if self.task == 'classification':
            y = self(x)
            self.reparameterize()
            z = self(x)
            print((y - z).abs().sum() / z.abs().sum())
        else:
            y = self(x)
            self.reparameterize()
            z = self(x)
            print([(i - j).abs().sum() / j.abs().sum() for i, j in zip(y, z)])


@register_model
def enformer_small(update_cfg: Dict[str, Any] = dict(crop_pct=0.9), **kwargs):
    model = EnsembleFormer(arch='S', **kwargs)  # drop_path_rate=(0.1, 0.1)
    cfg = DEFAULT_CFG
    cfg.update(update_cfg)
    model.pretrained_cfg = cfg
    return model

@register_model
def enformer_small_deploy(update_cfg: Dict[str, Any] = dict(crop_pct=0.9), **kwargs):
    model = EnsembleFormer(arch='S', deploy=True, **kwargs)  # drop_path_rate=(0.1, 0.1)
    cfg = DEFAULT_CFG
    cfg.update(update_cfg)
    model.pretrained_cfg = cfg
    return model


@register_model
def enformer_base(update_cfg: Dict[str, Any] = dict(crop_pct=0.9), **kwargs):
    model = EnsembleFormer(arch='B', **kwargs)  # drop_path_rate=(0.1, 0.1)
    cfg = DEFAULT_CFG
    cfg.update(update_cfg)
    model.pretrained_cfg = cfg
    return model

@register_model
def enformer_base_deploy(update_cfg: Dict[str, Any] = dict(crop_pct=0.9), **kwargs):
    model = EnsembleFormer(arch='B', deploy=True, **kwargs)  # drop_path_rate=(0.1, 0.1)
    cfg = DEFAULT_CFG
    cfg.update(update_cfg)
    model.pretrained_cfg = cfg
    return model


@register_model
def enformer_large(update_cfg: Dict[str, Any] = dict(crop_pct=0.9), **kwargs):
    model = EnsembleFormer(arch='L', **kwargs)  # drop_path_rate=(0.1, 0.1)
    cfg = DEFAULT_CFG
    cfg.update(update_cfg)
    model.pretrained_cfg = cfg
    return model

@register_model
def enformer_large_deploy(update_cfg: Dict[str, Any] = dict(crop_pct=0.9), **kwargs):
    model = EnsembleFormer(arch='L', deploy=True, **kwargs)  # drop_path_rate=(0.1, 0.1)
    cfg = DEFAULT_CFG
    cfg.update(update_cfg)
    model.pretrained_cfg = cfg
    return model


if __name__ == '__main__':
    imgs = torch.rand(4, 3, 224, 224).cuda()
    enformer = enformer_small().cuda()
    enformer_deploy = enformer_small_deploy().cuda()
    print(enformer(imgs).shape)
    enformer._test_equivalency(imgs)

    n_parameters = sum(p.numel() for p in enformer.parameters() if p.requires_grad)
    print("number of params: {:.2f}M".format(n_parameters / 1024 ** 2))
    get_flops(enformer)

    n_parameters = sum(p.numel() for p in enformer_deploy.parameters() if p.requires_grad)
    print("number of params: {:.2f}M".format(n_parameters / 1024 ** 2))
    get_flops(enformer_deploy)

    throughput = 0
    avg_throughput = 0
    iters = 3
    for i in range(iters):
        throughput = compute_throughput(enformer, 256, 224)
        avg_throughput += throughput
        print(throughput)
    print("avg throughput: ", avg_throughput / iters)

    throughput = 0
    avg_throughput = 0
    iters = 3
    for i in range(iters):
        throughput = compute_throughput(enformer_deploy, 256, 224)
        avg_throughput += throughput
        print(throughput)
    print("avg throughput: ", avg_throughput / iters)
