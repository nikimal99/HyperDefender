import numpy as np
import torch
from scipy.sparse import coo_matrix
from hype.hyla_utils import aug_normalized_adjacency, sparse_mx_to_torch_sparse_tensor


def load_perturbed_data(data_path, ptb_lvl):
    labels_path = f'{data_path}/labels.npy'
    feat_path = f'{data_path}/features.npy'
    idx_train_path = f'{data_path}/idx_train.npy'
    idx_test_path = f'{data_path}/idx_test.npy'
    idx_val_path = f'{data_path}/idx_val.npy'
    adj_path = f'{data_path}/adj{ptb_lvl}.npy'

    dense_adj_matrix = np.load(adj_path, allow_pickle=True)
    feat_matrix = np.load(feat_path, allow_pickle=True)
    idx_train = np.load(idx_train_path, allow_pickle=True).tolist()
    idx_test = np.load(idx_test_path, allow_pickle=True).tolist()
    idx_val = np.load(idx_val_path, allow_pickle=True).tolist()

    labels = np.load(labels_path)
    labels = torch.tensor(labels, dtype=torch.long)

    sparse_adj_matrix = coo_matrix(dense_adj_matrix)

    coo_indices = torch.tensor(feat_matrix.nonzero())
    coo_values = torch.from_numpy(feat_matrix[feat_matrix.nonzero()])
    size = tuple(feat_matrix.shape)

    feat_matrix_sparse = torch.sparse_coo_tensor(coo_indices, coo_values, size=size, dtype=torch.double)

    data = {}

    adj_n = aug_normalized_adjacency(sparse_adj_matrix)
    data['adj_train'] = sparse_mx_to_torch_sparse_tensor(adj_n)

    data['features'] = feat_matrix_sparse
    data['labels'] = labels
    data['idx_train'] = idx_train
    data['idx_test'] = idx_test
    data['idx_val'] = idx_val
    return data