#!/usr/bin/env python3s
from torch.optim.optimizer import Optimizer, required
import torch
import scipy.sparse as sp
import numpy as np
import tensorly as tl


class RiemannianSGD(Optimizer):
    r"""Riemannian stochastic gradient descent.
    Args:
        rgrad (Function): Function to compute the Riemannian gradient
           from the Euclidean gradient
        retraction (Function): Function to update the retraction
           of the Riemannian gradient
    """

    def __init__(
            self,
            params,
            lr=required,
            rgrad=required,
            expm=required,
    ):
        defaults = {
            'lr': lr,
            'rgrad': rgrad,
            'expm': expm,
        }
        super(RiemannianSGD, self).__init__(params, defaults)

    def step(self, lr=None, counts=None, **kwargs):
        """Performs a single optimization step.
        Arguments:
            lr (float, optional): learning rate for the current update.
        """
        loss = None

        for group in self.param_groups:
            for p in group['params']:
                lr = lr or group['lr']
                rgrad = group['rgrad']
                expm = group['expm']

                if p.grad is None:
                    continue
                d_p = p.grad.data
                # make sure we have no duplicates in sparse tensor
                if d_p.is_sparse:
                    d_p = d_p.coalesce()
                d_p = rgrad(p.data, d_p)
                d_p.mul_(-lr)
                expm(p.data, d_p)
        return loss


class PGD(Optimizer):
    def __init__(self, params, proxs, alphas, lr=required, momentum=0, dampening=0, weight_decay=0):
        defaults = dict(lr=lr, momentum=0, dampening=0,
                        weight_decay=0, nesterov=False)

        self.proxs = proxs
        self.alphas = alphas

        super(PGD, self).__init__(params, defaults)

        for group in self.param_groups:
            group.setdefault('proxs', proxs)
            group.setdefault('alphas', alphas)

    def __setstate__(self, state):
        super(PGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('proxs', self.proxs)
            group.setdefault('alphas', self.alphas)

    def step(self, delta=0, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            proxs = group['proxs']
            alphas = group['alphas']

            # apply the proximal operator to each parameter in a group
            for param in group['params']:
                for prox_operator, alpha in zip(proxs, alphas):
                    # param.data.add_(lr, -param.grad.data)
                    # param.data.add_(delta)
                    param.data = prox_operator(param.data, alpha=alpha*lr)


class ProxOperators():
    """
    Proximal Operators.
    """

    def __init__(self):
        self.nuclear_norm = None

    def prox_l1(self, data, alpha):
        """Proximal operator for l1 norm.
        """
        data = torch.mul(torch.sign(data), torch.clamp(
            torch.abs(data)-alpha, min=0))
        return data

    def prox_nuclear(self, data, alpha):
        """Proximal operator for nuclear norm (trace norm).
        """
        device = data.device
        U, S, V = np.linalg.svd(data.cpu())
        U, S, V = torch.FloatTensor(U).to(device), torch.FloatTensor(
            S).to(device), torch.FloatTensor(V).to(device)
        self.nuclear_norm = S.sum()

        diag_S = torch.diag(torch.clamp(S-alpha, min=0))
        return torch.matmul(torch.matmul(U, diag_S), V)

    def prox_nuclear_truncated_2(self, data, alpha, k=50):
        device = data.device
        tl.set_backend('pytorch')
        U, S, V = tl.truncated_svd(data.cpu(), n_eigenvecs=k)
        U, S, V = torch.FloatTensor(U).to(device), torch.FloatTensor(S).to(device), torch.FloatTensor(V).to(device)
        self.nuclear_norm = S.sum()

        S = torch.clamp(S-alpha, min=0)

        # make diag_S sparse matrix
        indices = torch.tensor((range(0, len(S)), range(0, len(S)))).to(device)
        values = S
        diag_S = torch.sparse.FloatTensor(indices, values, torch.Size((len(S), len(S))))
        V = torch.spmm(diag_S, V)
        V = torch.matmul(U, V)
        return V

    def prox_nuclear_truncated(self, data, alpha, k=50):
        device = data.device
        indices = torch.nonzero(data).t()
        # modify this based on dimensionality
        values = data[indices[0], indices[1]]
        data_sparse = sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()))
        U, S, V = sp.linalg.svds(data_sparse, k=k)
        U, S, V = torch.FloatTensor(U).to(device), torch.FloatTensor(S).to(device), torch.FloatTensor(V).to(device)
        self.nuclear_norm = S.sum()
        diag_S = torch.diag(torch.clamp(S-alpha, min=0))
        return torch.matmul(torch.matmul(U, diag_S), V)

    def prox_nuclear_cuda(self, data, alpha):
        device = data.device
        U, S, V = torch.svd(data)

        self.nuclear_norm = S.sum()
        S = torch.clamp(S-alpha, min=0)
        indices = torch.tensor([range(0, U.shape[0]), range(0, U.shape[0])]).to(device)
        values = S
        diag_S = torch.sparse.FloatTensor(indices, values, torch.Size(U.shape))

        V = torch.spmm(diag_S, V.t_())
        V = torch.matmul(U, V)
        return V
