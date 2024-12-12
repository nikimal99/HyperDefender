import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
import numpy as np
import timeit
from .hyla_utils import acc_f1
from .hyla_utils import PoissonKernel, sample_boundary, measure_tensor_size
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import gc


# ############3########## HyLa-GNN ################################
class HyLa(nn.Module):
    def __init__(self, manifold, dim, size, HyLa_fdim, scale=0.1, sparse=False, **kwargs):
        super(HyLa, self).__init__()
        self.manifold = manifold
        self.lt = manifold.allocate_lt(size, dim, sparse)
        self.manifold.init_weights(self.lt)
        self.dim = dim
        self.Lambdas = scale * torch.randn(HyLa_fdim)
        self.boundary = sample_boundary(HyLa_fdim, self.dim, cls='RandomUniform')
        self.bias = 2 * np.pi * torch.rand(HyLa_fdim)

    def forward(self):
        with torch.no_grad():
            e_all = self.manifold.normalize(self.lt.weight)
        PsK = PoissonKernel(e_all, self.boundary.to(e_all.device))
        angles = self.Lambdas.to(e_all.device)/2.0 * torch.log(PsK)
        eigs = torch.cos(angles + self.bias.to(e_all.device)) * torch.sqrt(PsK)**(self.dim-1)
        return eigs

    def optim_params(self):
        return [{
            'params': self.lt.parameters(),
            'rgrad': self.manifold.rgrad,
            'expm': self.manifold.expm,
            'logm': self.manifold.logm,
            'ptransp': self.manifold.ptransp,
        }]

class RFF(nn.Module):
    def __init__(self, manifold, dim, size, HyLa_fdim, scale=0.1, sparse=False, **kwargs):
        super(RFF, self).__init__()
        self.manifold = manifold
        self.lt = manifold.allocate_lt(size, dim, sparse)
        self.manifold.init_weights(self.lt)
        self.norm = 1. / np.sqrt(dim)
        self.Lambdas = nn.Parameter(torch.from_numpy(np.random.normal(loc=0, scale=scale, size=(dim, HyLa_fdim))), requires_grad=False)
        self.bias = nn.Parameter(torch.from_numpy(np.random.uniform(0, 2 * np.pi, size=HyLa_fdim)),requires_grad=False)

    def forward(self):
        with torch.no_grad():
            e_all = self.manifold.normalize(self.lt.weight)
        features = self.norm * np.sqrt(2) * torch.cos(e_all @ self.Lambdas + self.bias)
        return features

    def optim_params(self):
        return [{
            'params': self.lt.parameters(),
            'rgrad': self.manifold.rgrad,
            'expm': self.manifold.expm,
            'logm': self.manifold.logm,
            'ptransp': self.manifold.ptransp,
        }]

class SGC(nn.Module):
    def __init__(self, nfeat, nclass, opt):
        super(SGC, self).__init__()
        self.opt = opt
        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x, adj):
        self.adj = adj.to_dense().to(self.opt.device)
        self.precomputed_adj = adj.to_dense().to(self.opt.device)
        for i in range(self.opt.order - 1):
            self.precomputed_adj = torch.mm(self.precomputed_adj, self.adj)

        x = torch.mm(self.precomputed_adj, x)
        logits =  self.W(x)
        return F.log_softmax(logits, dim=1) #(CrossEntropy takes unnormalized logits)

class GCN(nn.Module):
    def __init__(self, nfeat, nclass, opt):
        super().__init__()

        self.opt = opt
        self.conv1 = GCNConv(nfeat, 16)
        self.conv2 = GCNConv(16, nclass)

    def forward(self, x, adj):
        if adj.is_sparse:
            adj = adj.coalesce().indices().to(self.opt.device)
        else:
            adj = adj.to_sparse().to(self.opt.device)
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj)
        return F.log_softmax(x, dim=1)

class sage(nn.Module):
    def __init__(self, nfeat, nclass, opt):
        super().__init__()
        self.opt = opt

        self.conv1 = SAGEConv(nfeat, 32)
        self.conv2 = SAGEConv(32, nclass)

    def forward(self, x, adj):
        adj = adj.coalesce().indices().to(self.opt.device)
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj)
        return F.log_softmax(x, dim=1)

class GAT(nn.Module):
    def __init__(self, nfeat, nclass, opt):
        super().__init__()
        self.opt = opt

        self.conv1 = GATConv(nfeat, 16)
        self.conv2 = GATConv(16, nclass)

    def forward(self, x, adj):
        adj = adj.coalesce().indices().to(self.opt.device)
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj)
        return F.log_softmax(x, dim=1)


class HylaGNNTrainer():
    def __init__(self, model_e, model_c, optimizer_e, optimizer_c, adj, opt, log):
        self.model_e = model_e
        self.model_c = model_c
        self.optimizer_e = optimizer_e
        self.optimizer_c = optimizer_c
        self.adj = adj
        self.opt = opt
        self.log = log

    def test_regression(self, data, test_labels, mask):
        with torch.no_grad():
            self.model_e.eval()
            self.model_c.eval()
            HyLa_features = self.model_e()
            if self.opt.use_feats:
                HyLa_features = torch.mm(data['features'].to(self.opt.device), HyLa_features)
            predictions = self.model_c(HyLa_features, self.adj)
            del HyLa_features
            acc, f1 = acc_f1(predictions[data[mask]], test_labels[data[mask]])
        if self.opt.metric == 'f1':
            return f1
        return acc

    def train(self, data, ckps=None):
        self.model_e.train()
        self.model_c.train()
        val_acc_best = 0.0
        train_acc_best = 0.0
        for epoch in range(self.opt.epoch_start, self.opt.epochs):
            t_start = timeit.default_timer()
            self.optimizer_e.zero_grad()
            self.optimizer_c.zero_grad()
            HyLa_features = self.model_e()
            if self.opt.use_feats:
                HyLa_features = torch.mm(data['features'].to(self.opt.device), HyLa_features)
            predictions = self.model_c(HyLa_features, self.adj)
            del HyLa_features

            loss = F.nll_loss(predictions[data["idx_train"]], data['labels'][data['idx_train']].to(self.opt.device))
            loss.backward()
            self.optimizer_e.step()
            self.optimizer_c.step()

            train_acc = self.test_regression(data, data['labels'].to(self.opt.device), "idx_train")
            val_acc = self.test_regression(data, data['labels'].to(self.opt.device), "idx_val")
            if val_acc > val_acc_best:
                val_acc_best = val_acc
                if ckps is not None:
                    ckps[0].save({
                        'model': self.model_e.state_dict(),
                        'epoch': epoch,
                        'val_acc_best': val_acc_best,
                    })
                    ckps[1].save({
                        'model': self.model_c.state_dict(),
                        'epoch': epoch,
                        'val_acc_best': val_acc_best,
                    })
            if train_acc > train_acc_best:
                train_acc_best = train_acc
            if self.opt.progress:
                self.log.info(
                    'running stats: {'
                    f'"epoch": {epoch}, '
                    f'"elapsed": {timeit.default_timer()-t_start:.2f}, '
                    f'"train_acc": {train_acc*100.0:.2f}%, '
                    f'"val_acc": {val_acc*100.0:.2f}%, '
                    f'"loss_c": {loss.cpu().item():.4f}, '
                    '}'
                )
            gc.collect()
            torch.cuda.empty_cache()
        return train_acc, train_acc_best, val_acc, val_acc_best


class originalGNNTrainer():
    def __init__(self, model_c, optimizer_e, optimizer_c, adj, opt, log):
        # self.model_e = model_e
        self.model_c = model_c
        # self.optimizer_e = optimizer_e
        self.optimizer_c = optimizer_c
        self.adj = adj
        self.opt = opt
        self.log = log

    def test_regression(self, data, test_labels, mask):
        with torch.no_grad():
            # self.model_e.eval()
            self.model_c.eval()
            # HyLa_features = self.model_e()
            # if self.opt.use_feats:
            #     HyLa_features = torch.mm(data['features'].to(self.opt.device), HyLa_features)
            predictions = self.model_c(data['features'], self.adj)
            # del HyLa_features
            acc, f1 = acc_f1(predictions[data[mask]], test_labels[data[mask]])
        if self.opt.metric == 'f1':
            return f1
        return acc

    def train(self, data, ckps=None):
        # self.model_e.train()
        self.model_c.train()
        val_acc_best = 0.0
        train_acc_best = 0.0
        for epoch in range(self.opt.epoch_start, self.opt.epochs):
            t_start = timeit.default_timer()
            # self.optimizer_e.zero_grad()
            self.optimizer_c.zero_grad()
            
            predictions = self.model_c(data['features'], self.adj)
            loss = F.nll_loss(predictions[data["idx_train"]], data['labels'][data['idx_train']].to(self.opt.device))
            loss.backward()
            # self.optimizer_e.step()
            self.optimizer_c.step()

            train_acc = self.test_regression(data, data['labels'].to(self.opt.device), "idx_train")
            val_acc = self.test_regression(data, data['labels'].to(self.opt.device), "idx_val")
            if val_acc > val_acc_best:
                val_acc_best = val_acc
                if ckps is not None:
                    ckps[0].save({
                        'model': self.model_c.state_dict(),
                        'epoch': epoch,
                        'val_acc_best': val_acc_best,
                    })
                    ckps[1].save({
                        'model': self.model_c.state_dict(),
                        'epoch': epoch,
                        'val_acc_best': val_acc_best,
                    })
            if train_acc > train_acc_best:
                train_acc_best = train_acc
            if self.opt.progress:
                self.log.info(
                    'running stats: {'
                    f'"epoch": {epoch}, '
                    f'"elapsed": {timeit.default_timer()-t_start:.2f}, '
                    f'"train_acc": {train_acc*100.0:.2f}%, '
                    f'"val_acc": {val_acc*100.0:.2f}%, '
                    f'"loss_c": {loss.cpu().item():.4f}, '
                    '}'
                )
            gc.collect()
            torch.cuda.empty_cache()
        return train_acc, train_acc_best, val_acc, val_acc_best



# ############### GNNGuard ####################################

import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, to_undirected
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    to_edge_index
)
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value

from typing import Optional
from .hyla_utils import glorot, zeros

def gcn_norm(
    edge_index: Adj,
    edge_weight: OptTensor = None,
    num_nodes: Optional[int] = None,
    improved: bool = False,
    add_self_loops: bool = True,
    flow: str = "source_to_target",
    dtype: Optional[torch.dtype] = None,
):
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'gcn_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        deg = scatter(value, col, 0, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

        return set_sparse_value(adj_t, value), None

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


class ModGCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, improved=False, cached=False, bias=True, normalize=True, **kwargs):
        super(ModGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.tensor(out_channels, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

        fill_value = 1 if not improved else 2
        # self-loop already added in the att_coef function"""

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))
        # edge_index = to_undirected(edge_index, x.size(0))  # add non-direct edges
        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, self.improved, x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


##ModGcn from gcn.py

import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
# from deeprobust.graph import utils
from copy import deepcopy
import scipy
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import numpy as np
import scipy.sparse as sp
# from deeprobust.graph.utils import *
from torch_geometric.nn import GINConv, GATConv, GCNConv, JumpingKnowledge
from torch.nn import Sequential, Linear, ReLU
from sklearn.preprocessing import normalize
# from deeprobust.graph.defense.basicfunction import att_coef
# from sklearn.metrics import f1_score
from scipy.sparse import lil_matrix

from . import hyla_utils


class GNNGuard(nn.Module):

    def __init__(self, nfeat, nclass, opt, nhid=32, dropout=0.5, lr=0.01, drop=False, weight_decay=5e-4, n_edge=1,with_relu=True,
                 with_bias=True, attention=True):

        super(GNNGuard, self).__init__()

        # assert device is not None, "Please specify 'device'!"
        self.opt = opt
        self.device = opt.device

        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.dropout = dropout
        self.lr = lr
        self.attention = attention
        weight_decay =0  # set weight_decay as 0

        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.n_edge = n_edge
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.gate = Parameter(torch.rand(1)).to(self.device) # creat a generator between [0,1]
        self.test_value = Parameter(torch.rand(1))
        self.drop_learn_1 = Linear(2, 1)
        self.drop_learn_2 = Linear(2, 1)
        self.drop = drop
        self.bn1 = torch.nn.BatchNorm1d(nhid)
        self.bn2 = torch.nn.BatchNorm1d(nhid)
        nclass = int(nclass)

        """GCN from geometric"""
        """network from torch-geometric, """
        self.gc1 = ModGCNConv(nfeat, nhid, bias=True,)
        self.gc2 = ModGCNConv(nhid, nclass, bias=True, )
        self.logits = None

        """GAT from torch-geometric"""
        # nclass = int(nclass)
        # self.gc1 = GATConv(nfeat, nhid, heads=8, dropout=0.6)
        # self.gc2 = GATConv(nhid*8, nclass, heads=1, concat=True, dropout=0.6)

        """GIN from torch-geometric"""
        # dim = 32
        # nn1 = Sequential(Linear(nfeat, dim), ReLU(), )
        # self.gc1 = GINConv(nn1)
        # # self.bn1 = torch.nn.BatchNorm1d(dim)
        # nn2 = Sequential(Linear(dim, dim), ReLU(), )
        # self.gc2 = GINConv(nn2)
        # self.jump = JumpingKnowledge(mode='cat')
        # # self.bn2 = torch.nn.BatchNorm1d(dim)
        # self.fc2 = Linear(dim, int(nclass))

        # """JK-Nets"""
        # num_features = nfeat
        # dim = 32
        # nn1 = Sequential(Linear(num_features, dim), ReLU(), )
        # self.gc1 = GINConv(nn1)
        # self.bn1 = torch.nn.BatchNorm1d(dim)
        #
        # nn2 = Sequential(Linear(dim, dim), ReLU(), )
        # self.gc2 = GINConv(nn2)
        # nn3 = Sequential(Linear(dim, dim), ReLU(), )
        # self.gc3 = GINConv(nn3)
        #
        # self.jump = JumpingKnowledge(mode='cat') # 'cat', 'lstm', 'max'
        # self.bn2 = torch.nn.BatchNorm1d(dim)
        # # self.fc1 = Linear(dim*3, dim)
        # self.fc2 = Linear(dim*2, int(nclass))

    def forward(self, x, adj_):
        """we don't change the edge_index, just update the edge_weight;
        some edge_weight are regarded as removed if it equals to zero"""
        x = x.to_dense().to(self.device)
        adj = adj_.to(self.device)
        """GCN and GAT"""
        if self.attention:
            adj = self.att_coef(x, adj, i=0)
        edge_index = adj._indices().to(self.device)
        x = self.gc1(x, edge_index, edge_weight=adj._values().to(self.device))
        x = F.relu(x)
        # x = self.bn1(x)
        if self.attention:  # if attention=True, use attention mechanism
            adj_2 = self.att_coef(x, adj, i=1).to(self.device)
            adj_memory = adj_2.to_dense()  # without memory
            adj_memory = self.gate * adj.to_dense().to(self.device) + (1 - self.gate) * adj_2.to_dense()
            row, col = adj_memory.nonzero()[:,0], adj_memory.nonzero()[:,1]
            edge_index = torch.stack((row, col), dim=0)
            adj_values = adj_memory[row, col]
        else:
            edge_index = adj._indices()
            adj_values = adj._values()
        self.logits = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index, edge_weight=adj_values)

        # self.logits = x
        # """GIN"""
        # if self.attention:
        #     adj = self.att_coef(x, adj, i=0)
        # x = F.relu(self.gc1(x, edge_index=edge_index, edge_weight=adj._values()))
        # if self.attention:  # if attention=True, use attention mechanism
        #     adj_2 = self.att_coef(x, adj, i=1)
        #     adj_values = self.gate * adj._values() + (1 - self.gate) * adj_2._values()
        # else:
        #     adj_values = adj._values()
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = F.relu(self.gc2(x, edge_index=edge_index, edge_weight=adj_values))
        # # x = [x] ### Add Jumping        # x = self.jump(x)
        # x = F.dropout(x, p=0.2,training=self.training)
        # x = self.fc2(x)

        # """JK-Nets"""
        # if self.attention:
        #     adj = self.att_coef(x, adj, i=0)
        # x1 = F.relu(self.gc1(x, edge_index=edge_index, edge_weight=adj._values()))
        # if self.attention:  # if attention=True, use attention mechanism
        #     adj_2 = self.att_coef(x1, adj, i=1)
        #     adj_values = self.gate * adj._values() + (1 - self.gate) * adj_2._values()
        # else:
        #     adj_values = adj._values()
        # x1 = F.dropout(x1, self.dropout, training=self.training)
        # x2 = F.relu(self.gc2(x1, edge_index=edge_index, edge_weight=adj_values))
        # x2 = F.dropout(x2, self.dropout, training=self.training)
        # x_last = self.jump([x1, x2])
        # x_last = F.dropout(x_last, self.dropout,training=self.training)
        # x = self.fc2(x_last)

        return F.log_softmax(x, dim=1)

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
        self.drop_learn_1.reset_parameters()
        self.drop_learn_2.reset_parameters()
        try:
            self.gate.reset_parameters()
            self.fc2.reset_parameters()
        except:
            pass

    def att_coef(self, fea, edge_index, is_lil=False, i=0):
        if is_lil == False:
            edge_index = edge_index._indices()
        else:
            edge_index = edge_index.tocoo()

        n_node = fea.shape[0]
        row, col = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]

        fea_copy = fea.cpu().data.numpy()
        sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)  # try cosine similarity
        sim = sim_matrix[row, col]
        sim[sim<0.1] = 0
        # print('dropped {} edges'.format(1-sim.nonzero()[0].shape[0]/len(sim)))

        # """use jaccard for binary features and cosine for numeric features"""
        # fea_start, fea_end = fea[edge_index[0]], fea[edge_index[1]]
        # isbinray = np.array_equal(fea_copy, fea_copy.astype(bool))  # check is the fea are binary
        # np.seterr(divide='ignore', invalid='ignore')
        # if isbinray:
        #     fea_start, fea_end = fea_start.T, fea_end.T
        #     sim = jaccard_score(fea_start, fea_end, average=None)  # similarity scores of each edge
        # else:
        #     fea_copy[np.isinf(fea_copy)] = 0
        #     fea_copy[np.isnan(fea_copy)] = 0
        #     sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)  # try cosine similarity
        #     sim = sim_matrix[edge_index[0], edge_index[1]]
        #     sim[sim < 0.01] = 0

        """build a attention matrix"""
        att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
        att_dense[row, col] = sim
        if att_dense[0, 0] == 1:
            att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format="lil")
        # normalization, make the sum of each row is 1
        att_dense_norm = normalize(att_dense, axis=1, norm='l1')


        """add learnable dropout, make character vector"""
        if self.drop:
            character = np.vstack((att_dense_norm[row, col].A1,
                                     att_dense_norm[col, row].A1))
            character = torch.from_numpy(character.T)
            drop_score = self.drop_learn_1(character)
            drop_score = torch.sigmoid(drop_score)  # do not use softmax since we only have one element
            mm = torch.nn.Threshold(0.5, 0)
            drop_score = mm(drop_score)
            mm_2 = torch.nn.Threshold(-0.49, 1)
            drop_score = mm_2(-drop_score)
            drop_decision = drop_score.clone().requires_grad_()
            # print('rate of left edges', drop_decision.sum().data/drop_decision.shape[0])
            drop_matrix = lil_matrix((n_node, n_node), dtype=np.float32)
            drop_matrix[row, col] = drop_decision.cpu().data.numpy().squeeze(-1)
            att_dense_norm = att_dense_norm.multiply(drop_matrix.tocsr())  # update, remove the 0 edges

        if att_dense_norm[0, 0] == 0:  # add the weights of self-loop only add self-loop at the first layer
            degree = (att_dense_norm != 0).sum(1).A1
            lam = 1 / (degree + 1) # degree +1 is to add itself
            self_weight = sp.diags(np.array(lam), offsets=0, format="lil")
            att = att_dense_norm + self_weight  # add the self loop
        else:
            att = att_dense_norm

        row, col = att.nonzero()
        att_adj = np.vstack((row, col))
        att_edge_weight = att[row, col]
        att_edge_weight = np.exp(att_edge_weight)   # exponent, kind of softmax
        att_edge_weight = torch.tensor(np.array(att_edge_weight)[0], dtype=torch.float32)#.cuda()
        att_adj = torch.tensor(att_adj, dtype=torch.int64)#.cuda()

        shape = (n_node, n_node)
        new_adj = torch.sparse.FloatTensor(att_adj, att_edge_weight, shape)
        return new_adj

    def add_loop_sparse(self, adj, fill_value=1):
        # make identify sparse tensor
        row = torch.range(0, int(adj.shape[0]-1), dtype=torch.int64)
        i = torch.stack((row, row), dim=0)
        v = torch.ones(adj.shape[0], dtype=torch.float32)
        shape = adj.shape
        I_n = torch.sparse.FloatTensor(i, v, shape)
        return adj + I_n.to(self.device)

    def fit(self, features, adj, labels, idx_train, idx_val=None, idx_test=None, train_iters=81, att_0=None,
            attention=False, model_name=None, initialize=True, verbose=False, normalize=False, patience=510, ):
        '''
            train the gcn model, when idx_val is not None, pick the best model
            according to the validation loss
        '''
        self.sim = None
        self.idx_test = idx_test
        self.attention = attention
        # if self.attention:
        #     att_0 = self.att_coef_1(features, adj)
        #     adj = att_0 # update adj
        #     self.sim = att_0 # update att_0

        # self.device = self.gc1.weight.device

        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = hyla_utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        # normalize = False # we don't need normalize here, the norm is conducted in the GCN (self.gcn1) model
        # if normalize:
        #     if utils.is_sparse_tensor(adj):
        #         adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        #     else:
        #         adj_norm = utils.normalize_adj_tensor(adj)
        # else:
        #     adj_norm = adj
        # add self loop
        adj = self.add_loop_sparse(adj)


        """The normalization gonna be done in the GCNConv"""
        self.adj_norm = adj
        self.features = features
        self.labels = labels

        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            if patience < train_iters:
                self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
            else:
                self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train], weight=None)   # this weight is the weight of each training nodes
            loss_train.backward()
            optimizer.step()
            if verbose and i % 20 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            # print('epoch', i)
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            self.eval()

            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = hyla_utils.accuracy(output[idx_val], labels[idx_val])
            # acc_test = utils.accuracy(output[self.idx_test], labels[self.idx_test])

            # if verbose and i % 5 == 0:
            #     print('Epoch {}, training loss: {}, val acc: {}, '.format(i, loss_train.item(), acc_val))

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
        # """my test"""
        # output_ = self.forward(self.features, self.adj_norm)
        # acc_test_ = utils.accuracy(output_[self.idx_test], labels[self.idx_test])
        # print('With best weights, test acc:', acc_test_)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.features, self.adj_norm)

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))


            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)

    def test(self, idx_test):
        self.eval()
        output = self.predict()  # here use the self.features and self.adj_norm in training stage
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = hyla_utils.accuracy(output[idx_test], self.labels[idx_test])
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test, output

    def _set_parameters(self):
        # TODO
        pass

    def predict(self, features=None, adj=None):
        '''By default, inputs are unnormalized data'''
        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = hyla_utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if hyla_utils.is_sparse_tensor(adj):
                self.adj_norm = hyla_utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = hyla_utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)




