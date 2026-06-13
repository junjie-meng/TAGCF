import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix, bmat
import scipy.sparse as sp
from config.configurator import configs
from data_utils.datasets_general_cf import PairwiseTrnData, PairwiseWEpochFlagTrnData, AllRankTstData
import torch as t
import torch.utils.data as data
import os

class DataHandlerGeneralAGCF:
    def __init__(self):
        if configs['data']['name'] == 'amazon':
            predir = './data/amazon/'
        elif configs['data']['name'] == 'yelp':
            predir = './data/yelp/'
        elif configs['data']['name'] == 'steam':
            predir = './data/steam/'
        else:
            # raise NotImplementedError
            predir = './data/' + configs['data']['name'] + '/'
        self.trn_file = predir + 'trn_mat.pkl'
        self.val_file = predir + 'val_mat.pkl'
        self.tst_file = predir + 'tst_mat.pkl'
        self.attr_file = predir + 'attr_edges.pkl'


    def _load_one_mat(self, file):
        """Load one single adjacent matrix from file

        Args:
            file (string): path of the file to load

        Returns:
            scipy.sparse.coo_matrix: the loaded adjacent matrix
        """
        with open(file, 'rb') as fs:
            mat = (pickle.load(fs) != 0).astype(np.float32)
        if type(mat) != coo_matrix:
            mat = coo_matrix(mat)
        return mat
    
    def _normalize_adj(self, mat):
        """Laplacian normalization for mat in coo_matrix

        Args:
            mat (scipy.sparse.coo_matrix): the un-normalized adjacent matrix

        Returns:
            scipy.sparse.coo_matrix: normalized adjacent matrix
        """
        degree = np.array(mat.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()
    
    def _use_pre_norm(self):
        data_cfg = configs.get('data', {})
        return data_cfg.get('pre_norm', data_cfg.get('prenorm', True))

    def _make_torch_adj(self, mat, self_loop=False):
        """Transform uni-directional adjacent matrix in coo_matrix into bi-directional adjacent matrix in torch.sparse.FloatTensor

        Args:
            mat (coo_matrix): the uni-directional adjacent matrix

        Returns:
            torch.sparse.FloatTensor: the bi-directional matrix
        """
        if not self_loop:
            a = csr_matrix((configs['data']['user_num'], configs['data']['user_num']))
            b = csr_matrix((configs['data']['item_num'], configs['data']['item_num']))
        else:
            data = np.ones(configs['data']['user_num'])
            row_indices = np.arange(configs['data']['user_num'])
            column_indices = np.arange(configs['data']['user_num'])
            a = csr_matrix((data, (row_indices, column_indices)), shape=(configs['data']['user_num'], configs['data']['user_num']))

            data = np.ones(configs['data']['item_num'])
            row_indices = np.arange(configs['data']['item_num'])
            column_indices = np.arange(configs['data']['item_num'])
            b = csr_matrix((data, (row_indices, column_indices)), shape=(configs['data']['item_num'], configs['data']['item_num']))

        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = self._normalize_adj(mat)

        # make torch tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).to(configs['device'])
    
    def _make_uai_adj(self, trn_mat, attr_edges, self_loop=False):
        user_num, item_num = trn_mat.shape
        user_item_num, attr_num = attr_edges.shape
        trn_mat_csr = sp.csr_matrix(trn_mat) # num_user, num_item = trn_mat_csr.shape
        attr_csr = sp.csr_matrix(attr_edges)
        user_attribute_matrix = attr_csr[:user_num, :]
        item_attribute_matrix = attr_csr[user_num:, :]
        if not self_loop:
            a = csr_matrix((user_num, user_num))
            b = csr_matrix((item_num, item_num))
            c = csr_matrix((attr_num, attr_num))
        else:
            a = sp.eye(user_num, format='csr')
            b = sp.eye(item_num, format='csr')
            c = sp.eye(attr_num, format='csr')
        # pdb.set_trace()
        adj_mat = bmat([[a, trn_mat_csr, user_attribute_matrix],
                        [trn_mat_csr.T, b, item_attribute_matrix],# num_user+num_item+num_attr
                        [user_attribute_matrix.T, item_attribute_matrix.T, c]], format='coo')
        adj_mat = (adj_mat !=0 ) *1.0
        if self._use_pre_norm():
            adj_mat = self._normalize_adj(adj_mat)
        idxs = t.from_numpy(np.vstack([adj_mat.row, adj_mat.col]).astype(np.int64))
        vals = t.from_numpy(adj_mat.data.astype(np.float32))
        shape = t.Size(adj_mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).to(configs['device'])
    
    def _make_ui_adj(self, trn_mat, attr_edges, self_loop=False):
        user_num, item_num = trn_mat.shape
        _, attr_num = attr_edges.shape
        trn_mat_csr = sp.csr_matrix(trn_mat) # num_user, num_item = trn_mat_csr.shape
        if not self_loop:
            a = csr_matrix((user_num, user_num))
            b = csr_matrix((item_num, item_num))
            c = csr_matrix((attr_num, attr_num))
            d = csr_matrix((user_num, attr_num))
            e = csr_matrix((item_num, attr_num))

        else:
            a = sp.eye(user_num, format='csr')
            b = sp.eye(item_num, format='csr')
            c = sp.eye(attr_num, format='csr')
            d = sp.eye(user_num, attr_num, format='csr')
            e = sp.eye(item_num, attr_num, format='csr')
        adj_mat = bmat([[a, trn_mat_csr, d],
                        [trn_mat_csr.T, b, e],# num_user+num_item+num_attr
                        [d.T, e.T, c]], format='coo')
        adj_mat = (adj_mat !=0 ) *1.0
        if self._use_pre_norm():
            adj_mat = self._normalize_adj(adj_mat)
        idxs = t.from_numpy(np.vstack([adj_mat.row, adj_mat.col]).astype(np.int64))
        vals = t.from_numpy(adj_mat.data.astype(np.float32))
        shape = t.Size(adj_mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).to(configs['device'])
    
    def load_data(self):
        trn_mat = self._load_one_mat(self.trn_file)
        val_mat = self._load_one_mat(self.val_file)
        tst_mat = self._load_one_mat(self.tst_file)
        self.trn_mat = trn_mat
        configs['data']['user_num'], configs['data']['item_num'] = trn_mat.shape 
        if os.path.exists(self.attr_file):
            atr_mat = self._load_one_mat(self.attr_file)
            configs['data']['user_item_num'], configs['data']['attr_num'] = atr_mat.shape
            assert configs['data']['user_item_num'] == (configs['data']['user_num'] + configs['data']['item_num'])
        
        if configs['data']['mat_type'] == 'uai':
            self.torch_adj = self._make_uai_adj(trn_mat, atr_mat, self_loop=True)
            self.ui_adj = self._make_ui_adj(trn_mat, atr_mat, self_loop=True)
        else:
            self.torch_adj = self._make_torch_adj(trn_mat, self_loop=True)
        
        if configs['model']['name'] == 'gccf':
            self.torch_adj = self._make_torch_adj(trn_mat, self_loop=True)

        if configs['train']['loss'] == 'pairwise':
            trn_data = PairwiseTrnData(trn_mat)
        elif configs['train']['loss'] == 'pairwise_with_epoch_flag':
            trn_data = PairwiseWEpochFlagTrnData(trn_mat)

        val_data = AllRankTstData(val_mat, trn_mat)
        tst_data = AllRankTstData(tst_mat, trn_mat)
        self.test_dataloader = data.DataLoader(tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        self.valid_dataloader = data.DataLoader(val_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        self.train_dataloader = data.DataLoader(trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)