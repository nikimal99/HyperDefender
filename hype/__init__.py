#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from . import manifolds
import argparse
from . import models
import torch

MANIFOLDS = {
    'lorentz': manifolds.LorentzManifold,
    'poincare': manifolds.PoincareManifold,
    'euclidean': manifolds.EuclideanManifold,
}

MODELS = {
    'hyla': models.HyLa,
    'rff': models.RFF,
}

def build_model(opt, N):
    if isinstance(opt, argparse.Namespace):
        opt = vars(opt)
    manifold = MANIFOLDS[opt['manifold']](K=None)
    return MODELS[opt['model']](
        manifold,
        dim=opt['he_dim'],
        size=N,
        HyLa_fdim=opt['hyla_dim'],
        scale=opt['lambda_scale'],
        sparse=opt['sparse'],
    )

def get_model(model_opt, nfeat, nclass, opt):
    if model_opt == "SGC":
        model = models.SGC(nfeat=nfeat, nclass=nclass, opt=opt)
    elif model_opt == "GCN":
        model = models.GCN(nfeat=nfeat, nclass=nclass, opt=opt)
    elif model_opt == "sage":
        model = models.sage(nfeat=nfeat, nclass=nclass, opt=opt)
    elif model_opt == "GAT":
        model = models.GAT(nfeat=nfeat, nclass=nclass, opt=opt)
    elif model_opt == "GNNGuard":
        model = models.GNNGuard(nfeat=nfeat, nclass=nclass, opt=opt)
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))
    return model
