"""MF, LightGCN, and XSimGCL backbones used by the public runner."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class BaseRecommender(nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int, norm: bool):
        super().__init__()
        if num_users < 1 or num_items < 1 or dim < 1:
            raise ValueError("num_users, num_items, and dim must be positive")
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.dim = int(dim)
        self.norm = bool(norm)

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def contrastive_loss(
        self,
        users: torch.Tensor,
        positive_anchors: torch.Tensor,
        all_user: torch.Tensor,
        all_item: torch.Tensor,
    ) -> torch.Tensor:
        return all_user.new_zeros(())


class MF(BaseRecommender):
    def __init__(self, num_users: int, num_items: int, dim: int = 64, norm: bool = True):
        super().__init__(num_users, num_items, dim, norm)
        self.embedding_user = nn.Embedding(num_users, dim)
        self.embedding_item = nn.Embedding(num_items, dim)

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        users = self.embedding_user.weight
        items = self.embedding_item.weight
        if self.norm:
            users = F.normalize(users, p=2, dim=1)
            items = F.normalize(items, p=2, dim=1)
        return users, items


class LightGCN(BaseRecommender):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        graph: torch.Tensor,
        dim: int = 64,
        layers: int = 2,
        norm: bool = True,
    ):
        super().__init__(num_users, num_items, dim, norm)
        if not graph.is_sparse:
            raise ValueError("LightGCN graph must be a sparse COO tensor")
        if layers < 1:
            raise ValueError("LightGCN requires at least one propagation layer")
        self.register_buffer("graph", graph.coalesce(), persistent=False)
        self.layers = int(layers)
        self.embedding_user = nn.Embedding(num_users, dim)
        self.embedding_item = nn.Embedding(num_items, dim)

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        all_embedding = torch.cat((self.embedding_user.weight, self.embedding_item.weight))
        propagated = [all_embedding]
        for _ in range(self.layers):
            all_embedding = torch.sparse.mm(self.graph, all_embedding)
            propagated.append(all_embedding)
        output = torch.stack(propagated, dim=1).mean(dim=1)
        users, items = torch.split(output, (self.num_users, self.num_items))
        if self.norm:
            users = F.normalize(users, p=2, dim=1)
            items = F.normalize(items, p=2, dim=1)
        return users, items


class XSimGCL(BaseRecommender):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        graph: torch.Tensor,
        dim: int = 64,
        layers: int = 3,
        norm: bool = True,
        cl_rate: float = 0.05,
        eps: float = 0.1,
        cl_temp: float = 0.1,
        cl_layer: int = 0,
    ):
        super().__init__(num_users, num_items, dim, norm)
        if not graph.is_sparse:
            raise ValueError("XSimGCL graph must be a sparse COO tensor")
        if layers < 1:
            raise ValueError("XSimGCL requires at least one propagation layer")
        if not 0 <= cl_layer < layers:
            raise ValueError("cl_layer must index one propagated layer")
        if cl_temp <= 0.0:
            raise ValueError("cl_temp must be positive")
        if cl_rate < 0.0 or eps < 0.0:
            raise ValueError("cl_rate and eps must be non-negative")

        self.register_buffer("graph", graph.coalesce(), persistent=False)
        self.layers = int(layers)
        self.cl_rate = float(cl_rate)
        self.eps = float(eps)
        self.cl_temp = float(cl_temp)
        self.cl_layer = int(cl_layer)
        self.embedding_user = nn.Embedding(num_users, dim)
        self.embedding_item = nn.Embedding(num_items, dim)
        nn.init.xavier_uniform_(self.embedding_user.weight)
        nn.init.xavier_uniform_(self.embedding_item.weight)
        self._contrastive_all: Optional[torch.Tensor] = None

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        all_embedding = torch.cat((self.embedding_user.weight, self.embedding_item.weight))
        propagated = []
        self._contrastive_all = None
        for layer in range(self.layers):
            all_embedding = torch.sparse.mm(self.graph, all_embedding)
            if self.training:
                random_noise = torch.rand_like(all_embedding)
                all_embedding = all_embedding + (
                    torch.sign(all_embedding)
                    * F.normalize(random_noise, dim=-1)
                    * self.eps
                )
            propagated.append(all_embedding)
            if layer == self.cl_layer:
                self._contrastive_all = all_embedding

        output = torch.stack(propagated, dim=1).mean(dim=1)
        users, items = torch.split(output, (self.num_users, self.num_items))
        if self.norm:
            users = F.normalize(users, p=2, dim=1)
            items = F.normalize(items, p=2, dim=1)
        return users, items

    @staticmethod
    def _info_nce(
        final_view: torch.Tensor,
        contrastive_view: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        logits = (final_view @ contrastive_view.T) / temperature
        return -torch.diag(F.log_softmax(logits, dim=1)).mean()

    def contrastive_loss(
        self,
        users: torch.Tensor,
        positive_anchors: torch.Tensor,
        all_user: torch.Tensor,
        all_item: torch.Tensor,
    ) -> torch.Tensor:
        if self._contrastive_all is None:
            raise RuntimeError("compute() must precede XSimGCL contrastive_loss()")
        cl_users, cl_items = torch.split(
            self._contrastive_all, (self.num_users, self.num_items)
        )
        if self.norm:
            cl_users = F.normalize(cl_users, p=2, dim=1)
            cl_items = F.normalize(cl_items, p=2, dim=1)

        unique_users = torch.unique(users.long())
        unique_items = torch.unique(positive_anchors.long())
        user_loss = self._info_nce(
            all_user[unique_users], cl_users[unique_users], self.cl_temp
        )
        item_loss = self._info_nce(
            all_item[unique_items], cl_items[unique_items], self.cl_temp
        )
        return self.cl_rate * (user_loss + item_loss)


def build_model(
    name: str,
    num_users: int,
    num_items: int,
    graph: Optional[torch.Tensor],
    dim: int = 64,
    norm: bool = True,
    lightgcn_layers: int = 2,
    xsimgcl_layers: int = 3,
    cl_rate: float = 0.05,
    eps: float = 0.1,
    cl_temp: float = 0.1,
    cl_layer: int = 0,
) -> BaseRecommender:
    normalized = name.lower()
    if normalized == "mf":
        return MF(num_users, num_items, dim=dim, norm=norm)
    if graph is None:
        raise ValueError("A graph backbone requires the normalized training graph")
    if normalized == "lightgcn":
        return LightGCN(
            num_users,
            num_items,
            graph=graph,
            dim=dim,
            layers=lightgcn_layers,
            norm=norm,
        )
    if normalized == "xsimgcl":
        return XSimGCL(
            num_users,
            num_items,
            graph=graph,
            dim=dim,
            layers=xsimgcl_layers,
            norm=norm,
            cl_rate=cl_rate,
            eps=eps,
            cl_temp=cl_temp,
            cl_layer=cl_layer,
        )
    raise ValueError("Unknown backbone: {}".format(name))
