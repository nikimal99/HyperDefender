#!/usr/bin/env python3
# import sys
# sys.path.append("..")
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
import torch
import logging
import argparse
import json
from hype.checkpoint import LocalCheckpoint
from hype.optimizer import RiemannianSGD
from hype.defence import *
from hype.perturbed import *
from hype.models import *
from hype import MANIFOLDS, MODELS, build_model, get_model
from hype.hyla_utils import *
import torch.nn.functional as F
import timeit
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from hyperbolicity import hyperbolicity_sample
import networkx as nx
from scipy.sparse import coo_matrix
from deeprobust.graph.defense import GCN
import time

import warnings
warnings.filterwarnings("ignore")

def generate_ckpt(opt, model, path):
    checkpoint = LocalCheckpoint(
            path,
            include_in_all={'conf' : vars(opt)},
            start_fresh=opt.fresh
        )
    # get state from checkpoint
    state = checkpoint.initialize({'epoch': 0, 'model': model.state_dict()})
    model.load_state_dict(state['model'])
    opt.epoch_start = state['epoch']
    return checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description='Train HyLa-GNN for node classification tasks')

    parser.add_argument('-checkpoint', action='store_true', default=False)
    parser.add_argument('-task', type=str, default='nc', help='learning task')
    parser.add_argument('-dataset', type=str, required=True, help='Dataset identifier [cora|disease_nc|pubmed|citeseer|reddit|airport]')
    parser.add_argument('-he_dim', type=int, default=2, help='Hyperbolic Embedding dimension')
    parser.add_argument('-hyla_dim', type=int, default=100, help='HyLa feature dimension')
    parser.add_argument('-manifold', type=str, default='poincare', choices=MANIFOLDS.keys(), help='model of hyperbolic space')
    parser.add_argument('-gnn_model', type=str, default='SGC', choices=['SGC', 'GCN', 'sage', 'GAT', 'GNNGuard'], help='model of GNN to be used')
    parser.add_argument('-model', type=str, default='hyla', choices=MODELS.keys(), help='feature model class, hyla|rff')

    parser.add_argument('-attack', type=str, default=None, choices=['random', 'mettack', 'rnd'], help='attack method')
    parser.add_argument('-defence', type=str, default=None, choices=['Rwl-GNN', 'Pro-HyLa'], help='defence method')
    parser.add_argument('-ptb_lvl', type=str, default=0.1, help='perturbation level, e.g., [0.0, 0.2, 0.4, 0.6, ..., 1.0]')
    parser.add_argument('-lr_e', type=float, default=0.1, help='Learning rate for hyperbolic embedding')
    parser.add_argument('-lr_c', type=float, default=0.1, help='Learning rate for the GNN module')
    parser.add_argument('-lr_a', type=float, default=0.1, help='Learning rate for the adjacency matrix updation in case of Rwl-GNN')
    parser.add_argument('-lr_adj', type=float, default=0.1, help='Learning rate for the adjacency matrix updation in case of Pro-GNN')


    parser.add_argument('-alpha', type=float, default=1, help='weight of Frobeius norm')
    parser.add_argument('-gamma', type=float, default=1, help='weight of GCN')
    parser.add_argument('-beta', type=float, default=0.1, help='weight of feature smoothing')
    parser.add_argument('-alpha_prognn', type=float, default=1, help='Prox operator alpha (l1-loss) in case of pro-gnn')
    parser.add_argument('-beta_prognn', type=float, default=0.1, help='Prox operator beta (nuclear-loss) of pro-gnn')
    parser.add_argument('-lambda_prognn', type=float, default=0.1, help='weight of feature smoothing of pro-gnn')
    parser.add_argument('-phi_prognn', type=float, default=0, help='symmetric loss of adjacency matrix of pro-gnn')
    parser.add_argument('-inner_steps', type=int, default=2, help='steps for inner optimization')
    parser.add_argument('-outer_steps', type=int, default=1, help='steps for outer optimization')



    parser.add_argument('-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('-strategy', type=int, default=0, help='Epochs of burn in, some advanced definition')
    parser.add_argument('-eval_each', type=int, default=1, help='Run evaluation every n-th epoch')
    parser.add_argument('-fresh', action='store_true', default=False, help='Override checkpoint')
    parser.add_argument('-debug', action='store_true', default=False, help='Print debuggin output')
    parser.add_argument('-gpu', default=0, type=int, help='Which GPU to run on (-1 for no gpu)')
    parser.add_argument('-seed', default=43, type=int, help='random seed')
    parser.add_argument('-sparse', default=True, action='store_true', help='Use sparse gradients for embedding table')
    parser.add_argument('-quiet', action='store_true', default=False)
    parser.add_argument('-lre_type', choices=['scale', 'constant'], default='constant')
    parser.add_argument('-optim_type', choices=['adam', 'sgd'], default='adam', help='optimizer used for the classification SGC model')
    parser.add_argument('-metric', choices=['acc', 'f1'], default='acc', help='what metrics to report')
    parser.add_argument('-lambda_scale', type=float, default=0.07, help='scale of lambdas when generating HyLa features')
    parser.add_argument('-inductive', action='store_true', default=False, help='inductive training, used for reddit.')
    parser.add_argument('-use_feats', action='store_true', default=False, help='whether embed in the feature level, otherwise node level')
    parser.add_argument('-tuned', action='store_true', default=False, help='whether use tuned hyper-parameters')
    parser.add_argument('-order', type=int, default=2, help='order of the SGC')
    parser.add_argument('-symmetric', action='store_true', default=False, help='whether use symmetric normalization')

    parser.add_argument('-original_rwl', action='store_true', default=False, help='whether use original rwl gnn defence mech without HyLa')
    parser.add_argument('-jaccard', action='store_true', default=False, help='Use if jaccard pre-processing')
    parser.add_argument('-original', action='store_true', default=False, help='Use if vanilla GNN model(GNNGuard, SGC, GCN, sage, GAT) without hyperbolic embedding')

    parser.add_argument('-dropout', type=float, default=0.5, help='dropout for gnnguard')


    opt = parser.parse_args()
    return opt

def main(opt):
    if opt.tuned:
        with open(f'{currentdir}/hyper_parameters_{opt.he_dim}d.json',) as f:
            hyper_parameters = json.load(f)[opt.dataset]
        opt.he_dim = hyper_parameters['he_dim']
        opt.hyla_dim = hyper_parameters['hyla_dim']
        opt.order = hyper_parameters['order']
        opt.lambda_scale = hyper_parameters['lambda_scale']
        # opt.lr_e = hyper_parameters['lr_e']
        # opt.lr_c = hyper_parameters['lr_c']
        # opt.epochs = hyper_parameters['epochs']

    opt.metric = 'f1' if opt.dataset =='reddit' else 'acc'
    opt.epoch_start = 0
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    opt.split_seed = opt.seed
    opt.progress = not opt.quiet

    # setup debugging and logigng
    log_level = logging.DEBUG if opt.debug else logging.INFO
    log = logging.getLogger('HyLa')
    logging.basicConfig(level=log_level, format='%(message)s', stream=sys.stdout)

    # set default tensor type
    torch.set_default_tensor_type('torch.DoubleTensor')
    # set device
    opt.device = torch.device(f'cuda:{opt.gpu}' if opt.gpu >= 0 and torch.cuda.is_available() else 'cpu')


    # DATA Loading
    data_path = f'{parentdir}/attacks/{opt.attack}/{opt.dataset}'
    if opt.dataset in ['cora', 'disease', 'airport']:
        data = load_perturbed_data(data_path, opt.ptb_lvl)
    else:
        raise NotImplementedError

    if opt.jaccard:
            labels = data['labels']
            features = data['features'].to_dense().to(opt.device).to(torch.float32)

            modified_adj = drop_dissimilar_edges(features , data['adj_train'].to_dense().to(opt.device).to(torch.float32))
            # modified_adj = modified_adj.to_dense()  # Convert to dense tensor
            modified_adj = modified_adj.toarray()
            features, modified_adj, labels = to_tensor(features, modified_adj, labels, device= opt.device)
            data['features']= features.to(torch.double)
            data['adj_train'] = (modified_adj).to(torch.double).to_sparse()
            data['labels'] = labels

    ### setup dataset parameters and setting
    if opt.use_feats or opt.inductive:
        if opt.progress:
            log.info(f'hyperbolic Laplacian features used in the feature level ...')
        feature_dim = data['features'].size(1)
    else:
        if opt.progress:
            log.info(f'hyperbolic Laplacian features used in the node level ...')
        feature_dim = data['adj_train'].size(1)

    if opt.progress:
        log.info(f'info about the data, training set size :{len(data["idx_train"])}, val size:{len(data["idx_val"])}, test size: {len(data["idx_test"])}')
        log.info(f'size of original feature matrix: {data["features"].size()}, number of classes {data["labels"].max().item()+1}')


    # build feature models and setup optimizers
    model_e = build_model(opt, feature_dim).to(opt.device)

    # Scale learning rate if lre_type is 'scale'
    if opt.lre_type == 'scale':
        opt.lr_e = opt.lr_e * len(data['idx_train'])

    # Set up optimizer for feature model based on the specified manifold
    if opt.manifold == 'euclidean':
        optimizer_e = torch.optim.SGD(model_e.parameters(), lr=opt.lr_e)
    elif opt.manifold == 'poincare':
        optimizer_e = RiemannianSGD(model_e.optim_params(), lr=opt.lr_e)

    # build classification GNN models and setup optimizers
    num_features = feature_dim if opt.original_rwl else opt.hyla_dim

    if not opt.original_rwl:
        num_features = feature_dim if opt.original else opt.hyla_dim
        model_c = get_model(model_opt=opt.gnn_model, nfeat=num_features, nclass=data['labels'].max().item()+1, opt=opt).to(opt.device)
    else:
        model_c = GCN(nfeat=data['features'].size(1), nclass=data['labels'].max().item()+1, nhid=16, device=opt.device).to(opt.device)

    if opt.optim_type == 'sgd':
        optimizer_c = torch.optim.SGD(model_c.parameters(), lr=opt.lr_c)
    elif opt.optim_type == 'adam':
        optimizer_c = torch.optim.Adam(model_c.parameters(), lr=opt.lr_c)#, weight_decay=1.0e-4)
    else:
        raise NotImplementedError

    ckps = None
    if opt.checkpoint:
        # setup checkpoint
        ckp_fm = generate_ckpt(opt, model_e, f'{currentdir}/datasets/' + opt.dataset + '/fm.pt')
        ckp_cm = generate_ckpt(opt, model_c, f'{currentdir}/datasets/' + opt.dataset + '/cm.pt')
        ckps = (ckp_fm, ckp_cm)

    t_start_all = timeit.default_timer()

    if opt.defence == 'Rwl-GNN':
        trainer = RwlGNNTrainer(model_e=model_e, model_c=model_c, optimizer_e=optimizer_e, optimizer_c=optimizer_c, opt=opt, log=log)
        val_acc, val_acc_best = trainer.fit(data["adj_train"].to_dense().to(opt.device), data['labels'].to(opt.device), data)
    elif opt.defence == 'Pro-HyLa':
        trainer = ProGNNTrainer(model_e=model_e, model_c=model_c, optimizer_e=optimizer_e, optimizer_c=optimizer_c, opt=opt, log=log)

        val_acc, val_acc_best = trainer.fit(features=data['features'].to_dense().to(opt.device),
                                            adj=data["adj_train"].to_dense().to(opt.device),
                                            labels=data['labels'].to(opt.device),
                                            idx_train = data['idx_train'],
                                            idx_val = data['idx_val'],
                                            opt=opt)
    elif opt.defence is None:
        if not opt.original:
            trainer = HylaGNNTrainer(model_e=model_e, model_c=model_c, optimizer_e=optimizer_e, optimizer_c=optimizer_c, adj=data['adj_train'], opt=opt, log=log)
            train_acc, train_acc_best, val_acc, val_acc_best = trainer.train(data, ckps=ckps)
        else:
            trainer = originalGNNTrainer(model_c=model_c, optimizer_e=optimizer_e, optimizer_c=optimizer_c, adj=data['adj_train'], opt=opt, log=log)
            train_acc, train_acc_best, val_acc, val_acc_best = trainer.train(data, ckps=ckps)
    else:
        raise ValueError(f'Unknown defence: {opt.defence}')

    if opt.progress:
        log.info(f'TOTAL ELAPSED: {timeit.default_timer()-t_start_all:.2f}')

    if opt.checkpoint and ckps is not None:
        state_fm = ckps[0].load()
        state_cm = ckps[1].load()
        model_e.load_state_dict(state_fm['model'])
        model_c.load_state_dict(state_cm['model'])
        if opt.progress:
            log.info(f'early stopping, loading from epoch: {state_fm["epoch"]} with val_acc_best: {state_fm["val_acc_best"]}')

    if opt.defence == 'Rwl-GNN':
        test_acc = trainer.test(data, data['labels'].to(opt.device))

        log.info(
            f'"|| last val_acc": {val_acc*100.0:.2f}%, '
            f'"|| best val_acc": {val_acc_best*100.0:.2f}%, '
            f'"|| test_acc": {test_acc*100.0:.2f}%.'
        )
    elif opt.defence == 'Pro-HyLa':
        test_acc = trainer.test(data["features"].to_dense().to(opt.device),
                                data['labels'].to(opt.device),
                                data['idx_test'],
                                data['adj_train'].to_dense().to(opt.device))
    elif opt.defence is None:
        test_acc = trainer.test_regression(data, data['labels'].to(opt.device),'idx_test')

        log.info(
            f'"|| last train_acc": {train_acc*100.0:.2f}%, '
            f'"|| best train_acc": {train_acc_best*100.0:.2f}%, '
            f'"|| last val_acc": {val_acc*100.0:.2f}%, '
            f'"|| best val_acc": {val_acc_best*100.0:.2f}%, '
            f'"|| test_acc": {test_acc*100.0:.2f}%.'
        )
    else:
        raise ValueError(f'Unknown defence: {opt.defence}')

    return val_acc_best, test_acc, 0



if __name__ == '__main__':
    opt = parse_args()
    val_acc, test_acc, hyp = main(opt)
    print("Test Acc : ", test_acc)