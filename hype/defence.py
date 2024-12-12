import time
import numpy as np
from copy import deepcopy
from .optimizer import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.utils import accuracy
from sklearn.metrics import accuracy_score, f1_score
import warnings

# ######################### ProHyLa #######################


def sgc_precompute(adj, features, degree):
    nonzero_perc = []
    if degree == 0:
        number_nonzero = (features != 0).sum().item()
        percentage = number_nonzero*1.0/features.size(0)/features.size(1)*100.0
        nonzero_perc.append("%.2f" % percentage)
        print('input order 0, return raw feature')
        return features, nonzero_perc
    for i in range(degree):
        features = torch.spmm(adj, features)
        number_nonzero = (features != 0).sum().item()
        percentage = number_nonzero*1.0/features.size(0)/features.size(1)*100.0
        nonzero_perc.append("%.2f" % percentage)
    return features, nonzero_perc


def acc_f1(output, labels, average='micro'):
    preds = output.max(1)[1].type_as(labels)
    if preds.is_cuda:
        preds = preds.cpu()
        labels = labels.cpu()
    accuracy = accuracy_score(preds, labels)
    f1 = f1_score(preds, labels, average=average)
    return accuracy, f1


class ProGNNTrainer:
    def __init__(self, model_e, model_c, optimizer_e, optimizer_c, opt, log):
        self.device = opt.device
        self.opt = opt

        self.best_val_acc = 0
        self.best_val_loss = 10

        self.model_e = model_e
        self.model_c = model_c

        self.optimizer_e = optimizer_e
        self.optimizer_c = optimizer_c

        self.best_graph = None
        self.weights_e = None
        self.weights_c = None
        self.estimator = None

        self.prox_operators = ProxOperators()

    def fit(self, features, adj, labels, idx_train, idx_val, opt):  # added opt as an argument
        opt = self.opt

        estimator = EstimateAdj(adj, symmetric=opt.symmetric, device=self.device).to(self.device)
        self.estimator = estimator
        self.optimizer_adj = optim.SGD(estimator.parameters(), momentum=0.9, lr=opt.lr_adj)

        self.optimizer_l1 = PGD(estimator.parameters(), proxs=[self.prox_operators.prox_l1], lr=opt.lr_adj, alphas=[opt.alpha_prognn])

        warnings.warn("If you find the nuclear proximal operator runs too slow, you can modify line 77 to use self.prox_operators.prox_nuclear_cuda instead of self.prox_operators.prox_nuclear to perform the proximal on GPU. See details in https://github.com/ChandlerBang/Pro-GNN/issues/1")
        self.optimizer_nuclear = PGD(estimator.parameters(), proxs=[self.prox_operators.prox_nuclear_cuda], lr=opt.lr_adj, alphas=[opt.beta_prognn])

        # Train model
        t_total = time.time()
        for epoch in range(opt.epochs):
            for i in range(int(opt.outer_steps)):
                acc_val, best_acc_val = self.train_adj(epoch, features, adj, labels, idx_train, idx_val, opt)

            for i in range(int(opt.inner_steps)):
                self.train_gcn(epoch, features, estimator.estimated_adj, labels, idx_train, idx_val, opt)

        print("Optimization Finished!")

        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        print("picking the best model according to validation performance")
        self.model_e.load_state_dict(self.weights_e)
        self.model_c.load_state_dict(self.weights_c)

        return acc_val, best_acc_val

    def train_gcn(self, epoch, features, adj, labels, idx_train, idx_val, opt):  # added opt argument!
        opt = self.opt
        estimator = self.estimator
        adj = estimator.normalize()
        t = time.time()
        self.model_e.train()
        self.model_c.train()
        self.optimizer_e.zero_grad()
        self.optimizer_c.zero_grad()
        HyLa_features = self.model_e()

        if self.opt.use_feats:
            HyLa_features = torch.mm(features.to(self.opt.device), HyLa_features)
        # predict using model_c (whose input is the output of hyla)
        predictions = self.model_c(HyLa_features, adj)
        del HyLa_features  # delete intermediate hyla features to free up memory

        loss_train = F.nll_loss(predictions[idx_train], labels[idx_train].to(opt.device))
        # Backpropagate the gradients and perform a step of optimization for both models
        loss_train.backward()
        self.optimizer_e.step()
        self.optimizer_c.step()
        acc_train, f1_train = acc_f1(predictions[idx_train], labels[idx_train].to(opt.device))

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model_e.eval()
        self.model_c.eval()
        # Obtain HyLa_features by applying model_e to the input features
        HyLa_features = self.model_e()
        if self.opt.use_feats:
            HyLa_features = torch.mm(features.to(self.opt.device), HyLa_features)
        # Make predictions using model_c on the transformed features
        predictions = self.model_c(HyLa_features, adj)
        del HyLa_features
        # Calculate accuracy and F1 score using the acc_f1 function
        acc, f1 = acc_f1(predictions[idx_val], labels[idx_val].to(opt.device))
        loss_val = F.nll_loss(predictions[idx_val], labels[idx_val].to(opt.device))
        acc_val, f1_val = acc_f1(predictions[idx_val], labels[idx_val].to(opt.device))

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = adj.detach()
            self.weights_e = deepcopy(self.model_e.state_dict())
            self.weights_c = deepcopy(self.model_c.state_dict())
            if opt.debug:
                print('\t=== saving current graph/gcn, best_val_acc: %s' %
                      self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = adj.detach()
            self.weights_e = deepcopy(self.model_e.state_dict())
            self.weights_c = deepcopy(self.model_c.state_dict())
            if opt.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' %
                      self.best_val_loss.item())

        # if opt.debug:

        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

        return acc_val, self.best_val_acc

    # added another argument (opt)
    def train_adj(self, epoch, features, adj, labels, idx_train, idx_val, opt):
        estimator = self.estimator
        opt = self.opt
        if opt.debug:
            print("\n=== train_adj ===")
        t = time.time()
        estimator.train()
        self.optimizer_adj.zero_grad()

        loss_l1 = torch.norm(estimator.estimated_adj, 1)
        loss_fro = torch.norm(estimator.estimated_adj - adj, p='fro')
        normalized_adj = estimator.normalize()

        if opt.lambda_prognn:
            loss_smooth_feat = self.feature_smoothing(
                estimator.estimated_adj, features)
        else:
            loss_smooth_feat = 0 * loss_l1

        self.model_e.eval()
        self.model_c.eval()
        # Obtain HyLa_features by applying model_e to the input features
        HyLa_features = self.model_e()
        if self.opt.use_feats:
            HyLa_features = torch.mm(features.to(self.opt.device), HyLa_features)
        # Make predictions using model_c on the transformed features
        output = self.model_c(HyLa_features, adj)
        del HyLa_features
        loss_gcn = F.nll_loss(output[idx_train], labels[idx_train].to(opt.device))
        acc_train, f1_train = acc_f1(output[idx_train], labels[idx_train])

        loss_symmetric = torch.norm(estimator.estimated_adj
                                    - estimator.estimated_adj.t(), p="fro")

        loss_diffiential = loss_fro + opt.gamma * loss_gcn + \
            opt.lambda_prognn * loss_smooth_feat + opt.phi_prognn * loss_symmetric

        loss_diffiential.backward()

        self.optimizer_adj.step()
        loss_nuclear = 0 * loss_fro
        if opt.beta_prognn != 0:
            self.optimizer_nuclear.zero_grad()
            self.optimizer_nuclear.step()
            loss_nuclear = self.prox_operators.nuclear_norm

        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()

        total_loss = loss_fro \
            + opt.gamma * loss_gcn \
            + opt.alpha_prognn * loss_l1 \
            + opt.beta_prognn * loss_nuclear \
            + opt.phi_prognn * loss_symmetric

        estimator.estimated_adj.data.copy_(torch.clamp(estimator.estimated_adj.data, min=0, max=1))

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        normalized_adj = estimator.normalize()
        new_features, nonzero_perc = sgc_precompute(
            normalized_adj, features, opt.order)  # normalize if adj is not normalized
        self.model_e.eval()
        self.model_c.eval()
        HyLa_features = self.model_e()
        if self.opt.use_feats:
            HyLa_features = torch.mm(features.to(self.opt.device), HyLa_features)
        output = self.model_c(HyLa_features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(opt.device))
        acc_val, f1_val = acc_f1(output[idx_val], labels[idx_val].to(opt.device))
        if (opt.debug):
          print('Epoch: {:04d}'.format(epoch+1),
                'acc_train: {:.4f}'.format(acc_train.item()),
                'loss_val: {:.4f}'.format(loss_val.item()),
                'acc_val: {:.4f}'.format(acc_val.item()),
                'time: {:.4f}s'.format(time.time() - t))

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = normalized_adj.detach()
            self.weights_e = deepcopy(self.model_e.state_dict())
            self.weights_c = deepcopy(self.model_c.state_dict())
            if opt.debug:
                print(f'\t=== saving current graph/gcn, best_val_acc: %s' %
                      self.best_val_acc.item())

        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = normalized_adj.detach()
            self.weights_e = deepcopy(self.model_e.state_dict())
            self.weights_c = deepcopy(self.model_c.state_dict())
            if opt.debug:
                print(f'\t=== saving current graph/gcn, best_val_loss: %s' %
                      self.best_val_loss.item())

        if opt.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_fro: {:.4f}'.format(loss_fro.item()),
                      'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                      'loss_feat: {:.4f}'.format(loss_smooth_feat.item()),
                      'loss_symmetric: {:.4f}'.format(loss_symmetric.item()),
                      'delta_l1_norm: {:.4f}'.format(torch.norm(estimator.estimated_adj-adj, 1).item()),
                      'loss_l1: {:.4f}'.format(loss_l1.item()),
                      'loss_total: {:.4f}'.format(total_loss.item()),
                      'loss_nuclear: {:.4f}'.format(loss_nuclear.item()))

        return acc_val, self.best_val_acc

    def test(self, features, labels, idx_test, adj):
        """Evaluate the performance of ProGNN on test set
        """
        print("\t=== testing ===")
        with torch.no_grad():
          # Set both model_e and model_c to evaluation mode
          self.model_e.eval()
          self.model_c.eval()
          # Obtain HyLa_features by applying model_e to the input features
          HyLa_features = self.model_e()
          if self.opt.use_feats:
                HyLa_features = torch.mm(features.to(self.opt.device), HyLa_features)
          # Make predictions using model_c on the transformed features
          predictions = self.model_c(HyLa_features, adj)
          numpy_array = HyLa_features.cpu().numpy()
          del HyLa_features
          # Calculate accuracy and F1 score using the acc_f1 function
          acc, f1 = acc_f1(predictions[idx_test], labels[idx_test])
        return acc

    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj)/2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        L = L.to(torch.float64)
        D = D.to(torch.float64)
        r_inv = r_inv.to(torch.float64)

        r_inv = r_inv + 1e-3
        r_inv = r_inv.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        # L = r_mat_inv @ L

        L = r_mat_inv @ L @ r_mat_inv
        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat


class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, adj, symmetric=False, device='cpu'):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj)
        self.symmetric = symmetric
        self.device = device

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):

        if self.symmetric:
            adj = (self.estimated_adj + self.estimated_adj.t())/2
        else:
            adj = self.estimated_adj

        normalized_adj = self._normalize(
            adj + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx




# ######################### RwlGNN #######################

class RwlGNNTrainer:
    def __init__(self, model_e, model_c, optimizer_e, optimizer_c, opt, log):
        self.opt = opt
        self.log = log
        self.device = opt.device
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_graph = None
        self.estimator = None

        self.model_e = model_e
        self.model_c = model_c

        self.weights_e = deepcopy(self.model_e.state_dict())
        self.weights_c = deepcopy(self.model_c.state_dict())

        self.optimizer_e = optimizer_e
        self.optimizer_c = optimizer_c

        self.valid_cost = []
        self.train_acc = []
        self.valid_acc = []

    def fit(self, adj, labels, data):
        opt = self.opt
        self.symmetric = opt.symmetric

        lr_adj = self.opt.lr_a

        adj = (adj.t() + adj)/2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L_noise = D - adj

        # INIT
        self.weight = self.Linv(L_noise)

        self.weight.requires_grad = True
        self.weight = self.weight.to(self.device)

        x_xT = torch.matmul(data['features'], data['features'].t()).to(self.device)

        t_total = time.time()

        c = self.Lstar(2*L_noise*opt.alpha - opt.beta * x_xT)

        self.sgl_opt = AdamOptimizer(self.weight, lr=lr_adj)

        for epoch in range(opt.epochs):
                for i in range(int(opt.outer_steps)):
                    val_acc, best_val_acc = self.train_specific(epoch, L_noise, labels, data, c)

                for i in range(int(opt.inner_steps)):
                    estimate_adj = self.A()
                    self.train_gcn(epoch, estimate_adj, labels, data)

        self.log.info("Optimization Finished!")
        self.log.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        self.log.info("picking the best model according to validation performance")
        self.model_e.load_state_dict(self.weights_e)
        self.model_c.load_state_dict(self.weights_c)

        return val_acc, best_val_acc

    def w_grad(self, alpha, c):
        with torch.no_grad():
            grad_f = self.Lstar(alpha*self.L()) - c
            return grad_f

    def train_specific(self, epoch, L_noise, labels, data, c):
        if self.opt.debug:
            self.log.info("\n=== train_adj ===")
        t = time.time()

        y = self.weight.clone().detach()
        y = y.to(self.device)
        y.requires_grad = True

        loss_fro = self.opt.alpha * torch.norm(self.L(y) - L_noise, p='fro')
        normalized_adj = self.normalize(y)
        loss_smooth_feat = self.opt.beta * self.feature_smoothing(self.A(y), data['features'].to(self.device))

        if not self.opt.original_rwl:
            HyLa_features = self.model_e()
            if self.opt.use_feats:
                HyLa_features = torch.mm(data['features'].to(self.opt.device), HyLa_features)
            output = self.model_c(HyLa_features, normalized_adj)
        else:
            # breakpoint()
            output = self.model_c(data['features'].float().to(self.opt.device), normalized_adj.float())

        loss_gcn = self.opt.gamma * F.nll_loss(output[data["idx_train"]], labels[data["idx_train"]])
        acc_train = accuracy(output[data["idx_train"]], labels[data["idx_train"]])

        # loss_diffiential = loss_fro + gamma*loss_gcn+opt.lambda_ * loss_smooth_feat


        gcn_grad = torch.autograd.grad(
            inputs=y,
            outputs=loss_gcn,
            grad_outputs=torch.ones_like(loss_gcn),
            only_inputs=True,
        )[0]

        sgl_grad = self.w_grad(self.opt.alpha, c)

        total_grad = sgl_grad + gcn_grad

        self.weight = self.sgl_opt.backward_pass(total_grad)
        self.weight = torch.clamp(self.weight, min=0)

        total_loss = loss_fro + loss_gcn + loss_smooth_feat

        self.model_e.eval()
        self.model_c.eval()

        normalized_adj = self.normalize()

        if not self.opt.original_rwl:
            HyLa_features = self.model_e()
            if self.opt.use_feats:
                HyLa_features = torch.mm(data['features'].to(self.opt.device), HyLa_features)
            output = self.model_c(HyLa_features, normalized_adj)
        else:
            output = self.model_c(data['features'].float().to(self.opt.device), normalized_adj.float())

        loss_val = F.nll_loss(output[data['idx_val']], labels[data['idx_val']])
        acc_val = accuracy(output[data['idx_val']], labels[data['idx_val']])

        self.log.info(
                    'running stats: {'
                    f'"epoch": {epoch}, '
                    f'"train_acc": {acc_train.item()*100.0:.2f}%, '
                    f'"val_acc": {acc_val.item()*100.0:.2f}%, '
                    f'"total_loss": {total_loss.item():.4f}, '
                    '}'
                )

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = normalized_adj.detach()
            self.weights_e = deepcopy(self.model_e.state_dict())
            self.weights_c = deepcopy(self.model_c.state_dict())
            if self.opt.debug:
                self.log.info(f'\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        # if loss_val < self.best_val_loss:
        #     self.best_val_loss = loss_val
        #     self.best_graph = normalized_adj.detach()
        #     self.weights_e = deepcopy(self.model_e.state_dict())
        #     self.weights_c = deepcopy(self.model_c.state_dict())
        #     if self.opt.debug:
        #         self.log.info(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())

        if self.opt.debug:
            if epoch % 1 == 0:
                self.log.info('Epoch: {:04d}'.format(epoch+1),
                      'loss_fro: {:.4f}'.format(loss_fro.item()),
                      'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                      'loss_feat: {:.4f}'.format(loss_smooth_feat.item()),
                      'loss_total: {:.4f}'.format(total_loss.item()))

        return acc_val, self.best_val_acc

    def train_gcn(self, epoch, adj, labels, data):
        opt = self.opt
        adj = self.normalize()

        t = time.time()
        self.model_c.train()
        self.optimizer_c.zero_grad()

        if not self.opt.original_rwl:
            self.model_e.train()
            self.optimizer_e.zero_grad()

            HyLa_features = self.model_e()
            if self.opt.use_feats:
                HyLa_features = torch.mm(data['features'].to(self.opt.device), HyLa_features)
            output = self.model_c(HyLa_features, adj)
        else:
            output = self.model_c(data['features'].float().to(self.opt.device), adj.float())

        loss_train = F.nll_loss(output[data['idx_train']], labels[data['idx_train']])
        acc_train = accuracy(output[data['idx_train']], labels[data['idx_train']])
        loss_train.backward(retain_graph=True)

        if not self.opt.original_rwl:
            self.optimizer_e.step()
        self.optimizer_c.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model_c.eval()

        if not self.opt.original_rwl:
            self.model_e.eval()
            HyLa_features = self.model_e()
            if self.opt.use_feats:
                HyLa_features = torch.mm(data['features'].to(self.opt.device), HyLa_features)
            output = self.model_c(HyLa_features, adj)
        else:
            output = self.model_c(data['features'].float().to(self.opt.device), adj.float())

        loss_val = F.nll_loss(output[data['idx_val']], labels[data['idx_val']])
        acc_val = accuracy(output[data['idx_val']], labels[data['idx_val']])

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = adj.detach()
            self.weights_e = deepcopy(self.model_e.state_dict())
            self.weights_c = deepcopy(self.model_c.state_dict())
            if opt.debug:
                print('\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        # if loss_val < self.best_val_loss:
        #     self.best_val_loss = loss_val
        #     self.best_graph = adj.detach()
        #     self.weights_e = deepcopy(self.model_e.state_dict())
        #     self.weights_c = deepcopy(self.model_c.state_dict())
        #     if opt.debug:
        #         print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())

        if opt.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      'time: {:.4f}s'.format(time.time() - t))

    def test(self, data, labels):
        """Evaluate the performance of RWL-GNN on test set
        """
        print("\t=== testing ===")
        self.model_c.eval()
        adj = self.best_graph

        if not self.opt.original_rwl:
            self.model_e.eval()
            HyLa_features = self.model_e()
            if self.opt.use_feats:
                HyLa_features = torch.mm(data['features'].to(self.opt.device), HyLa_features)
            output = self.model_c(HyLa_features, adj)
        else:
            output = self.model_c(data['features'].float().to(self.opt.device), adj.float())

        loss_test = F.nll_loss(output[data["idx_test"]], labels[data["idx_test"]])
        acc_test = accuracy(output[data["idx_test"]], labels[data["idx_test"]])

        self.log.info(
                    '\tTest set results:'
                    f'"test_acc": {acc_test.item()*100.0:.2f}%, '
                    f'"total_loss": {loss_test.item():.4f}, '
                    '}'
                )
        return acc_test.item()

    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj)/2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv + 1e-3
        r_inv = r_inv.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        # L = r_mat_inv @ L
        L = r_mat_inv @ L @ r_mat_inv

        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat

    def A(self, weight=None):
        # with torch.no_grad():
        if weight == None:
            k = self.weight.shape[0]
            a = self.weight
        else:
            k = weight.shape[0]
            a = weight
        n = int(0.5 * (1 + np.sqrt(1 + 8 * k)))
        Aw = torch.zeros((n, n), device=self.device)
        b = torch.triu_indices(n, n, 1)
        Aw[b[0], b[1]] = a
        Aw = Aw + Aw.t()
        return Aw

    def L(self, weight=None):
        if weight == None:
            k = len(self.weight)
            a = self.weight
        else:
            k = len(weight)
            a = weight
        n = int(0.5*(1 + np.sqrt(1+8*k)))
        Lw = torch.zeros((n, n), device=self.device)
        b = torch.triu_indices(n, n, 1)
        Lw[b[0], b[1]] = -a
        Lw = Lw + Lw.t()
        row, col = np.diag_indices_from(Lw)
        Lw[row, col] = -Lw.sum(axis=1)
        return Lw

    def Linv(self, M):
      with torch.no_grad():
        N = M.shape[0]
        k = int(0.5*N*(N-1))
        # l=0
        w = torch.zeros(k, device=self.device)
        # in the triu_indices try changing the 1 to 0/-1/2 for other
        # ascpect of result on how you want the diagonal to be included
        indices = torch.triu_indices(N, N, 1)
        M_t = torch.tensor(M)
        w = -M_t[indices[0], indices[1]]
        return w

    def Lstar(self, M):
        N = M.shape[1]
        k = int(0.5*N*(N-1))
        w = torch.zeros(k, device=self.device)
        tu_enteries = torch.zeros(k, device=self.device)
        tu = torch.triu_indices(N, N, 1)

        tu_enteries = M[tu[0], tu[1]]
        diagonal_enteries = torch.diagonal(M)

        b_diagonal = diagonal_enteries[0:N-1]
        x = torch.linspace(N-1, 1, steps=N-1, dtype=torch.long, device=self.device)
        x_r = x[:N]
        diagonal_enteries_a = torch.repeat_interleave(b_diagonal, x_r)

        new_arr = torch.tile(diagonal_enteries, (N, 1))
        tu_new = torch.triu_indices(N, N, 1)
        diagonal_enteries_b = new_arr[tu_new[0], tu_new[1]]
        w = diagonal_enteries_a+diagonal_enteries_b-2*tu_enteries

        return w

    def normalize(self, w=None):
        if self.symmetric:
            if w == None:
                adj = (self.A() + self.A().t())
            else:
                adj = self.A(w)

            adj = adj + adj.t()
        else:
            if w == None:
                adj = self.A()
            else:
                adj = self.A(w)

        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx


class AdamOptimizer:
    def __init__(self, weights, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = torch.zeros_like(weights)
        self.v = torch.zeros_like(weights)
        self.t = 0
        self.theta = weights

    def backward_pass(self, gradient):
        self.t = self.t + 1
        self.m = self.beta1*self.m + (1 - self.beta1)*gradient
        self.v = self.beta2*self.v + (1 - self.beta2)*(gradient**2)
        m_hat = self.m/(1 - self.beta1**self.t)
        v_hat = self.v/(1 - self.beta2**self.t)
        self.theta = self.theta - self.lr * (m_hat/(torch.sqrt(v_hat) + self.epsilon))
        return self.theta


class sgd:
    def __init__(self, weights, lr=1e-2):
        self.lr = lr
        self.theta = weights

    def backward_pass(self, gradient):
        self.theta = self.theta - self.lr * gradient
        return self.theta
