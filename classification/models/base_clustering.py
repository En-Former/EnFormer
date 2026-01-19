import builtins
import importlib
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'PartitionalClustering', 'FuzzyClustering', 'PossibilisticClustering', 'ProbabilisticClustering',
    'build_clustering', 'compute_similarity',
]


class PartitionalClustering(nn.Module):  # inspired by K-Means (one-iteration)
    def __init__(self, dim: int, num_clusters: int, return_clusters: bool = False,
                 with_cluster_bias: bool = True, with_cluster_position: bool = False,
                 use_cosine: bool = True, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_clusters = num_clusters
        self.return_clusters = return_clusters
        self.sim_alpha = nn.Parameter(torch.ones(num_clusters))
        self.sim_beta = nn.Parameter(torch.zeros(num_clusters))
        self.with_cluster_bias = with_cluster_bias
        self.with_cluster_position = with_cluster_position
        self.use_cosine = use_cosine
        self.scale = None if self.use_cosine else dim ** -0.5
        if self.with_cluster_bias:
            if self.with_cluster_position:
                self.clusters_proj = nn.Sequential(nn.Linear(self.dim + 2, self.dim), nn.SiLU())
                self.clusters_bias = nn.Parameter(torch.zeros(self.num_clusters, self.dim + 2))
                self.value_clusters_proj = nn.Sequential(nn.Linear(self.dim + 2, self.dim), nn.SiLU())
                self.value_clusters_bias = nn.Parameter(torch.zeros(self.num_clusters, self.dim + 2))
            else:
                self.clusters_proj = nn.Sequential(nn.Linear(self.dim, self.dim), nn.SiLU())
                self.clusters_bias = nn.Parameter(torch.zeros(self.num_clusters, self.dim))
                self.value_clusters_proj = nn.Sequential(nn.Linear(self.dim, self.dim), nn.SiLU())
                self.value_clusters_bias = nn.Parameter(torch.zeros(self.num_clusters, self.dim))

    def forward(self, points: torch.Tensor, clusters: torch.Tensor,
                values: torch.Tensor, value_clusters: torch.Tensor) -> torch.Tensor:
        if self.with_cluster_bias:
            clusters = self.clusters_proj(clusters + self.clusters_bias)  # [B, M, D]
            value_clusters = self.value_clusters_proj(value_clusters + self.value_clusters_bias)  # [B, M, D]

        similarity = compute_similarity(clusters, points, self.scale, self.use_cosine)  # [B, M, N]
        similarity_ = self.sim_alpha.view(1, -1, 1) * similarity + self.sim_beta.view(1, -1, 1)
        mask = F.gumbel_softmax(similarity_, tau=1.0, hard=True, dim=1)  # [B, M, N]
        assignment = torch.sigmoid(similarity_) * mask  # [B, M, N]
        aggregation = torch.baddbmm(value_clusters, assignment, values) / (mask.sum(dim=-1, keepdim=True) + 1.0)  # [B, M, D]

        if self.return_clusters:
            return aggregation

        if self.training and torch.is_autocast_enabled():  # avoid the loss becoming nan when amp-enabled
            x = (aggregation.permute(0, 2, 1).unsqueeze(-1) * assignment.unsqueeze(1)).sum(dim=2)  # [B, D, N]
        else:
            x = aggregation.permute(0, 2, 1) @ assignment  # [B, D, N]
        return x

    def forward_fast(self, points: torch.Tensor, clusters: torch.Tensor,
                     values: torch.Tensor, value_clusters: torch.Tensor) -> torch.Tensor:
        if self.with_cluster_bias:
            clusters = self.clusters_proj(clusters + self.clusters_bias)  # [B, M, D]
            value_clusters = self.value_clusters_proj(value_clusters + self.value_clusters_bias)  # [B, M, D]

        similarity = compute_similarity(clusters, points, self.scale, self.use_cosine)  # [B, M, N]
        similarity_ = self.sim_alpha.view(1, -1, 1) * similarity + self.sim_beta.view(1, -1, 1)
        max_assign_idx = similarity_.argmax(dim=1, keepdim=True)  # [b, 1, N]
        mask = torch.zeros_like(similarity_, device=similarity_.device).scatter_(1, max_assign_idx, 1.)  # [B, M, N]
        assignment = torch.sigmoid(similarity_) * mask  # [B, M, N]
        aggregation = torch.baddbmm(value_clusters, assignment, values) / (mask.sum(dim=-1, keepdim=True) + 1.0)  # [B, M, D]

        if self.return_clusters:
            return aggregation

        if self.training and torch.is_autocast_enabled():
            x = (aggregation.permute(0, 2, 1).unsqueeze(-1) * assignment.unsqueeze(1)).sum(dim=2)  # [B, D, N]
        else:
            x = aggregation.permute(0, 2, 1) @ assignment  # [B, D, N]
        return x


class FuzzyClustering(nn.Module):  # inspired by Fuzzy C-Means (one-iteration)
    def __init__(self, dim: int, num_clusters: int, return_clusters: bool = False,
                 with_cluster_bias: bool = True, with_cluster_position: bool = False,
                 use_cosine: bool = True, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_clusters = num_clusters
        self.return_clusters = return_clusters
        self.sim_alpha = nn.Parameter(torch.ones(num_clusters))
        self.sim_beta = nn.Parameter(torch.zeros(num_clusters))
        self.with_cluster_bias = with_cluster_bias
        self.with_cluster_position = with_cluster_position
        self.use_cosine = use_cosine
        self.scale = None if self.use_cosine else dim ** -0.5
        if self.with_cluster_bias:
            if self.with_cluster_position:
                self.clusters_proj = nn.Sequential(nn.Linear(self.dim + 2, self.dim), nn.SiLU())
                self.clusters_bias = nn.Parameter(torch.zeros(self.num_clusters, self.dim + 2))
                self.value_clusters_proj = nn.Sequential(nn.Linear(self.dim + 2, self.dim), nn.SiLU())
                self.value_clusters_bias = nn.Parameter(torch.zeros(self.num_clusters, self.dim + 2))
            else:
                self.clusters_proj = nn.Sequential(nn.Linear(self.dim, self.dim), nn.SiLU())
                self.clusters_bias = nn.Parameter(torch.zeros(self.num_clusters, self.dim))
                self.value_clusters_proj = nn.Sequential(nn.Linear(self.dim, self.dim), nn.SiLU())
                self.value_clusters_bias = nn.Parameter(torch.zeros(self.num_clusters, self.dim))

    def forward(self, points: torch.Tensor, clusters: torch.Tensor,
                values: torch.Tensor, value_clusters: torch.Tensor, eps=1e-7) -> torch.Tensor:
        if self.with_cluster_bias:
            clusters = self.clusters_proj(clusters + self.clusters_bias)  # [B, M, D]
            value_clusters = self.value_clusters_proj(value_clusters + self.value_clusters_bias)  # [B, M, D]

        similarity = compute_similarity(clusters, points, self.scale, self.use_cosine)  # [B, M, N]
        similarity_ = self.sim_alpha.view(1, -1, 1) * similarity + self.sim_beta.view(1, -1, 1)
        assignment = F.softmax(similarity_, dim=1)  # [B, M, N]
        aggregation = torch.baddbmm(value_clusters, assignment, values) / (assignment.sum(dim=-1, keepdim=True) + eps)  # [B, M, D]

        if self.return_clusters:
            return aggregation

        if self.training and torch.is_autocast_enabled():
            x = (aggregation.permute(0, 2, 1).unsqueeze(-1) * assignment.unsqueeze(1)).sum(dim=2)  # [B, D, N]
        else:
            x = aggregation.permute(0, 2, 1) @ assignment  # [B, D, N]
        return x


class PossibilisticClustering(nn.Module):  # inspired by Possibilistic C-Means (one-iteration)
    def __init__(self, dim: int, num_clusters: int, return_clusters: bool = False,
                 with_cluster_bias: bool = True, with_cluster_position: bool = False,
                 use_cosine: bool = True, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_clusters = num_clusters
        self.return_clusters = return_clusters
        self.sim_alpha = nn.Parameter(torch.ones(num_clusters))
        self.sim_beta = nn.Parameter(torch.zeros(num_clusters))
        self.with_cluster_bias = with_cluster_bias
        self.with_cluster_position = with_cluster_position
        self.use_cosine = use_cosine
        self.scale = None if self.use_cosine else dim ** -0.5
        if self.with_cluster_bias:
            if self.with_cluster_position:
                self.clusters_proj = nn.Sequential(nn.Linear(self.dim + 2, self.dim), nn.SiLU())
                self.clusters_bias = nn.Parameter(torch.zeros(self.num_clusters, self.dim + 2))
                self.value_clusters_proj = nn.Sequential(nn.Linear(self.dim + 2, self.dim), nn.SiLU())
                self.value_clusters_bias = nn.Parameter(torch.zeros(self.num_clusters, self.dim + 2))
            else:
                self.clusters_proj = nn.Sequential(nn.Linear(self.dim, self.dim), nn.SiLU())
                self.clusters_bias = nn.Parameter(torch.zeros(self.num_clusters, self.dim))
                self.value_clusters_proj = nn.Sequential(nn.Linear(self.dim, self.dim), nn.SiLU())
                self.value_clusters_bias = nn.Parameter(torch.zeros(self.num_clusters, self.dim))

    def forward(self, points: torch.Tensor, clusters: torch.Tensor,
                values: torch.Tensor, value_clusters: torch.Tensor, eps=1e-7) -> torch.Tensor:
        if self.with_cluster_bias:
            clusters = self.clusters_proj(clusters + self.clusters_bias)  # [B, M, D]
            value_clusters = self.value_clusters_proj(value_clusters + self.value_clusters_bias)  # [B, M, D]

        similarity = compute_similarity(clusters, points, self.scale, self.use_cosine)  # [B, M, N]
        similarity_ = self.sim_alpha.view(1, -1, 1) * similarity + self.sim_beta.view(1, -1, 1)
        assignment = torch.sigmoid(similarity_)  # [B, M, N]
        aggregation = torch.baddbmm(value_clusters, assignment, values) / (assignment.sum(dim=-1, keepdim=True) + eps)  # [B, M, D]

        if self.return_clusters:
            return aggregation

        if self.training and torch.is_autocast_enabled():
            x = (aggregation.permute(0, 2, 1).unsqueeze(-1) * assignment.unsqueeze(1)).sum(dim=2)  # [B, D, N]
        else:
            x = aggregation.permute(0, 2, 1) @ assignment  # [B, D, N]
        return x


class ProbabilisticClustering(nn.Module):  # Simplified Gaussian Mixture Model (one-iteration: E-step only)
    def __init__(self, dim: int, num_clusters: int, return_clusters: bool = False,
                 with_cluster_bias: bool = True, with_cluster_position: bool = False,
                 use_cosine: bool = True, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_clusters = num_clusters
        self.return_clusters = return_clusters
        self.sim_alpha = nn.Parameter(torch.ones(num_clusters))
        self.sim_beta = nn.Parameter(torch.zeros(num_clusters))
        self.log_priors = nn.Parameter(torch.zeros(self.num_clusters))
        self.log_vars = nn.Parameter(torch.zeros(self.num_clusters, self.dim))
        self.with_cluster_bias = with_cluster_bias
        self.with_cluster_position = with_cluster_position
        self.use_cosine = use_cosine
        self.scale = None if self.use_cosine else dim ** -0.5
        if self.with_cluster_bias:
            if self.with_cluster_position:
                self.clusters_proj = nn.Sequential(nn.Linear(self.dim + 2, self.dim), nn.SiLU())
                self.clusters_bias = nn.Parameter(torch.zeros(self.num_clusters, self.dim + 2))
                self.value_clusters_proj = nn.Sequential(nn.Linear(self.dim + 2, self.dim), nn.SiLU())
                self.value_clusters_bias = nn.Parameter(torch.zeros(self.num_clusters, self.dim + 2))
            else:
                self.clusters_proj = nn.Sequential(nn.Linear(self.dim, self.dim), nn.SiLU())
                self.clusters_bias = nn.Parameter(torch.zeros(self.num_clusters, self.dim))
                self.value_clusters_proj = nn.Sequential(nn.Linear(self.dim, self.dim), nn.SiLU())
                self.value_clusters_bias = nn.Parameter(torch.zeros(self.num_clusters, self.dim))

    def forward(self, points: torch.Tensor, clusters: torch.Tensor,
                values: torch.Tensor, value_clusters: torch.Tensor, eps=1e-7) -> torch.Tensor:
        if self.with_cluster_bias:
            clusters = self.clusters_proj(clusters + self.clusters_bias)  # [B, M, D]
            value_clusters = self.value_clusters_proj(value_clusters + self.value_clusters_bias)  # [B, M, D]

        points_exp = points.permute(0, 2, 1).unsqueeze(1)  # [B, 1, N, D]
        clusters_exp = clusters.unsqueeze(2)  # [B, M, 1, D]

        log_priors = self.log_priors.unsqueeze(0).unsqueeze(-1)  # [1, M, 1]
        log_vars = self.log_vars.unsqueeze(0)  # [1, M, D]

        precision = torch.exp(-log_vars).unsqueeze(2)  # [1, M, 1, D]
        diff_sq = (points_exp - clusters_exp).pow(2)  # [B, M, N, D]
        mahalanobis_sq = (diff_sq * precision).sum(dim=-1)  # [B, M, N]

        log_det_sigma = torch.sum(log_vars, dim=-1, keepdim=True)  # [1, M, 1]
        log_likelihood = -0.5 * (mahalanobis_sq + log_det_sigma)  # [B, M, N]
        log_likelihood_ = self.sim_alpha.view(1, -1, 1) * log_likelihood + self.sim_beta.view(1, -1, 1)

        responsibility = F.softmax(log_likelihood_ + log_priors, dim=1)  # [B, M, N]
        aggregation = torch.baddbmm(value_clusters, responsibility, values) / (responsibility.sum(dim=-1, keepdim=True) + eps)  # [B, M, D]

        if self.return_clusters:
            return aggregation

        if self.training and torch.is_autocast_enabled():
            x = (aggregation.permute(0, 2, 1).unsqueeze(-1) * responsibility.unsqueeze(1)).sum(dim=2)  # [B, D, N]
        else:
            x = aggregation.permute(0, 2, 1) @ responsibility  # [B, D, N]
        return x


# ---------------------------------------------------- experimental ----------------------------------------------------
class GMMClustering(nn.Module):  # Simplified Gaussian Mixture Model (multi-iteration: E-step + M-step)
    def __init__(self, dim: int, num_clusters: int, return_clusters: bool = False,
                 with_cluster_bias: bool = True, with_cluster_position: bool = False,
                 use_cosine: bool = True, **kwargs):
        super().__init__()


# -------------------------------------------------------- utils -------------------------------------------------------
def compute_similarity(x1: torch.Tensor, x2: torch.Tensor,
                       scale: Optional[float] = 1.0, use_cosine: bool = True) -> torch.Tensor:
    # [..., M, D] @ [..., D, N] -> [..., M, N]
    def compute_cosine_similarity(_x1: torch.Tensor, _x2: torch.Tensor) -> torch.Tensor:
        return F.normalize(_x1, dim=-1) @ F.normalize(_x2, dim=-2)

    def compute_dot_product(_x1: torch.Tensor, _x2: torch.Tensor, _scale: float = 1.0) -> torch.Tensor:
        return (_x1 @ _x2) * _scale

    if use_cosine:
        return compute_cosine_similarity(x1, x2)
    else:
        return compute_dot_product(x1, x2, scale)


def build_clustering(class_path: str, *args, **kwargs) -> nn.Module:
    if '.' not in class_path:
        if class_path in globals():
            cls = globals()[class_path]
        elif hasattr(builtins, class_path):
            cls = getattr(builtins, class_path)
        else:
            raise ValueError(f"class {class_path} not found in globals or builtins")
    else:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

    return cls(*args, **kwargs)
