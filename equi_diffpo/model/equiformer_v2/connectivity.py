# Modified from Diffusion-EDF
from typing import Tuple, Optional
import torch
from torch_cluster import radius_graph, radius, fps
from torch_scatter import scatter_add
from torch_cluster import knn


class RadiusGraph(torch.nn.Module):
    def __init__(self, r: float, max_num_neighbors: int):
        super().__init__()
        self.r: float = r
        self.max_num_neighbors: int = max_num_neighbors

    def forward(self, node_coord_src: torch.Tensor, node_feature_src: torch.Tensor, batch_src: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert node_coord_src.ndim == 2 and node_coord_src.shape[-1] == 3

        node_coord_dst = node_coord_src
        batch_dst = batch_src
        node_feature_dst = node_feature_src
        N_nodes = len(node_coord_dst)

        edge = radius_graph(node_coord_dst, r=self.r, batch=batch_dst, loop=False,
                            max_num_neighbors=self.max_num_neighbors)
        edge_dst = edge[0]
        edge_src = edge[1]
        degree = scatter_add(src=torch.ones_like(edge_dst), index=edge_dst, dim=0, dim_size=N_nodes)

        return node_feature_dst, node_coord_dst, edge_src, edge_dst, degree, batch_dst


class RadiusConnect(torch.nn.Module):
    def __init__(self, r: float, max_num_neighbors: int, offset: Optional[float] = None):
        super().__init__()
        self.r: float = r
        self.max_num_neighbors: int = max_num_neighbors
        if offset is not None:
            raise NotImplementedError
        self.offset = offset

    def forward(self, node_coord_src: torch.Tensor, batch_src: torch.Tensor, node_coord_dst: torch.Tensor,
                batch_dst: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        edge = radius(x=node_coord_src, y=node_coord_dst, r=self.r, batch_x=batch_src, batch_y=batch_dst,
                      max_num_neighbors=self.max_num_neighbors)
        edge_dst = edge[0]
        edge_src = edge[1]

        return edge_src, edge_dst


class FpsPool(torch.nn.Module):
    def __init__(self, ratio: float, random_start: bool, r: float, max_num_neighbors: int):
        super().__init__()
        self.ratio: float = ratio
        self.random_start: bool = random_start
        self.r: float = r
        self.max_num_neighbors: int = max_num_neighbors
        self.radius_connect: RadiusConnect = RadiusConnect(r=self.r, max_num_neighbors=self.max_num_neighbors)


    def forward(self, node_coord_src: torch.Tensor, batch_src: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list]:
        assert node_coord_src.ndim == 2 and node_coord_src.shape[-1] == 3
        if self.ratio != 1.0:
            node_dst_idx = fps(src=node_coord_src, batch=batch_src, ratio=self.ratio, random_start=self.random_start)
            node_dst_idx = torch.unique(node_dst_idx)
        else:
            node_dst_idx = torch.arange(0, len(node_coord_src), device=node_coord_src.device)
        # torch_cluster.fps returns a list of indices for each batch, but sometimes the indices are not unique

        node_coord_dst = node_coord_src.index_select(index=node_dst_idx, dim=0)
        batch_dst = batch_src.index_select(index=node_dst_idx, dim=0)
        N_nodes = len(node_dst_idx)

        edge_src, edge_dst = self.radius_connect(node_coord_src=node_coord_src, node_coord_dst=node_coord_dst,
                                                 batch_src=batch_src, batch_dst=batch_dst)

        non_self_idx = (node_dst_idx[edge_dst] != edge_src).nonzero().squeeze(-1)
        edge_src = edge_src[non_self_idx]
        edge_dst = edge_dst[non_self_idx]

        degree = scatter_add(src=torch.ones_like(edge_dst), index=edge_dst, dim=0, dim_size=N_nodes)

        return node_coord_dst, edge_src, edge_dst, degree, batch_dst, node_dst_idx


class URandPool(torch.nn.Module):
    '''
    Uniform Random Pool
    '''
    def __init__(self, ratio: float, random_start: bool, r: float, max_num_neighbors: int):
        super().__init__()
        self.ratio: float = ratio
        self.random_start: bool = random_start
        self.r: float = r
        self.max_num_neighbors: int = max_num_neighbors
        self.radius_connect: RadiusConnect = RadiusConnect(r=self.r, max_num_neighbors=self.max_num_neighbors)

    def forward(self, node_coord_src: torch.Tensor, batch_src: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list]:
        assert node_coord_src.ndim == 2 and node_coord_src.shape[-1] == 3
        if self.ratio != 1.0:
            node_dst_idx = torch.multinomial(torch.ones_like(node_coord_src[:, 0]), int(len(node_coord_src)*self.ratio))
            node_dst_idx = torch.unique(node_dst_idx)
        else:
            node_dst_idx = torch.arange(0, len(node_coord_src), device=node_coord_src.device)
        # torch_cluster.fps returns a list of indices for each batch, but sometimes the indices are not unique

        node_coord_dst = node_coord_src.index_select(index=node_dst_idx, dim=0)
        batch_dst = batch_src.index_select(index=node_dst_idx, dim=0)
        N_nodes = len(node_dst_idx)

        edge_src, edge_dst = self.radius_connect(node_coord_src=node_coord_src, node_coord_dst=node_coord_dst,
                                                 batch_src=batch_src, batch_dst=batch_dst)

        non_self_idx = (node_dst_idx[edge_dst] != edge_src).nonzero().squeeze(-1)
        edge_src = edge_src[non_self_idx]
        edge_dst = edge_dst[non_self_idx]

        degree = scatter_add(src=torch.ones_like(edge_dst), index=edge_dst, dim=0, dim_size=N_nodes)

        return node_coord_dst, edge_src, edge_dst, degree, batch_dst, node_dst_idx

class FpsKnnPool(torch.nn.Module):
    def __init__(self, ratio: float, random_start: bool, r: float, max_num_neighbors: int):
        super().__init__()
        self.ratio: float = ratio
        self.random_start: bool = random_start
        self.r: float = r
        self.max_num_neighbors: int = max_num_neighbors
        # self.radius_connect = RadiusConnect(r=self.r, max_num_neighbors=self.max_num_neighbors)

    def forward(self, node_coord_src: torch.Tensor, batch_src: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list]:
        assert node_coord_src.ndim == 2 and node_coord_src.shape[-1] == 3
        if self.ratio != 1.0:
            node_dst_idx = fps(src=node_coord_src, batch=batch_src, ratio=self.ratio, random_start=self.random_start)
            node_dst_idx = torch.unique(node_dst_idx)
        else:
            node_dst_idx = torch.arange(0, len(node_coord_src), device=node_coord_src.device)
        # torch_cluster.fps returns a list of indices for each batch, but sometimes the indices are not unique

        node_coord_dst = node_coord_src.index_select(index=node_dst_idx, dim=0)
        batch_dst = batch_src.index_select(index=node_dst_idx, dim=0)
        N_nodes = len(node_dst_idx)

        edge_dst, edge_src = knn(node_coord_src, node_coord_dst, self.max_num_neighbors,
                                 batch_x=batch_src, batch_y=batch_dst)

        non_self_idx = (node_dst_idx[edge_dst] != edge_src).nonzero().squeeze(-1)
        edge_src = edge_src[non_self_idx]
        edge_dst = edge_dst[non_self_idx]

        degree = scatter_add(src=torch.ones_like(edge_dst), index=edge_dst, dim=0, dim_size=N_nodes)

        return node_coord_dst, edge_src, edge_dst, degree, batch_dst, node_dst_idx


class AdaptiveOriginPool(torch.nn.Module):
    def __init__(self, random_start: bool, r: float, max_num_neighbors: int):
        super().__init__()
        self.random_start: bool = random_start
        self.r: float = r
        self.max_num_neighbors: int = max_num_neighbors
        self.radius_connect = RadiusConnect(r=self.r, max_num_neighbors=self.max_num_neighbors)

    def forward(self, node_coord_src: torch.Tensor, batch_src: torch.Tensor, origion=None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list]:
        assert node_coord_src.ndim == 2 and node_coord_src.shape[-1] == 3
        # torch_cluster.fps returns a list of indices for each batch, but sometimes the indices are not unique
        bs = batch_src.max()+1
        if origion is None:
            node_coord_dst = torch.zeros((bs, 3), device=node_coord_src.device)
            node_dst_idx = torch.arange(bs, device=node_coord_src.device)
            batch_dst = node_dst_idx
        else:
            node_coord_dst = origion.reshape(-1, 3)
            node_dst_idx = torch.arange(node_coord_dst.shape[0], device=node_coord_src.device)
            batch_dst = torch.arange(bs, device=node_coord_src.device).repeat_interleave(origion.shape[1], dim=0)

        edge_src, edge_dst = self.radius_connect(node_coord_src=node_coord_src, node_coord_dst=node_coord_dst,
                                                 batch_src=batch_src, batch_dst=batch_dst)

        # degree = scatter_add(src=torch.ones_like(edge_dst), index=edge_dst, dim=0, dim_size=bs)
        degree = None

        return node_coord_dst, edge_src, edge_dst, degree, batch_dst, node_dst_idx
