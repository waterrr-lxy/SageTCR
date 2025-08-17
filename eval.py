import torch
from torch import nn
import torch.nn.functional as F
from torch import randperm
from torch_geometric.nn import SAGEConv, GraphNorm, global_mean_pool, SAGPooling, global_max_pool
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from typing import Any
from scipy.sparse import coo_matrix
# from libauc.losses import AUCMLoss  # AUC loss
# from libauc.optimizers import PESG  # 为了优化AUC loss的一个优化器
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import os
import sys
import configparser

print('cuda可用性：', torch.cuda.is_available())


class PairData(Data):  # 主要是用于返回邻接矩阵的维度？
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_atom':
            return self.x_atom.size(0)
        
        if key == 'edge_index_res':
            return self.x_res.size(0)
        
        return super().__inc__(key, value)
    
    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:

        # 似乎是用来定义矩阵的拼接方式
        if key == 'y':  # 除了node_matrix和edge_index之外的自定义属性
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)
    
    
class SageTCRDataset(Dataset):
    def __init__(self, root, data_df, transform=None, pre_transform=None, pre_filter=None):
        
        '''
        root: dataset本地目录，下面会自动创建raw和processed两个子目录
        feature_type: atom或res，若传入atom则构建gab_dataset，若传入res则构建grb_dataset
        data_df: 传入的记录文件信息的dataframe，初次处理数据集的时候没关系，
                 如果读取数据集来训练模型请在外面shuffle好再传入
        '''
        self.data_df = data_df
        self.complex_ls = data_df['complex']
        
        # 自定义的初始化参数一定要写在super之前
        super(SageTCRDataset, self).__init__(root, transform, pre_transform, pre_filter)
     
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        '''
        这个模块的目的是检查数据集中的文件是否是处理过的，如果已经处理过久直接读取，不再处理了
        方法是根据返回值的列表检查这些文件在processed目录中是否都存在，所以返回值要放所有处理后的文件名
        '''
        return [f'{prefix}.pt' for prefix in self.complex_ls]
    
    def download(self):
        pass

    def process(self):
        '''
        很抽象的功能，可能是内部逻辑有一点问题，不支持读取初始化好的self.xxx变量，
        但是反而可以在不声明global的情况下直接调用外面的全局变量
        【管它了先这样用着吧】
        '''
        for idx, row in self.data_df.iterrows():

            # 从dataframe里面读文件位置信息
            prefix = row['complex']
            file_path = row['npz_path']
            label = row['label']

            # 读取文件
            npz_file = np.load(file_path)
            atom_node = npz_file['atom_node']
            atom_adj = npz_file['atom_adj']
            res_node = npz_file['res_node']
            res_adj = npz_file['res_adj']

            atom_node = torch.FloatTensor(atom_node)
            res_node = torch.FloatTensor(res_node)

            # 将邻接矩阵转化为稀疏矩阵：主要是为了节省存储空间
            # 因为GNN forward里面要求传入sparsetensor类型的邻接矩阵，所以需要预处理一下
            atom_edge_index = torch.LongTensor(np.vstack([coo_matrix(atom_adj).row, coo_matrix(atom_adj).col]))
            atom_edge_index = torch.LongTensor(atom_edge_index)
            res_edge_index = torch.LongTensor(np.vstack([coo_matrix(res_adj).row, coo_matrix(res_adj).col]))
            res_edge_index = torch.LongTensor(res_edge_index)
#             data = Data(x=x, edge_index=edge_index, y=torch.LongTensor([label]))  # 使用CrossEtropyloss
            data = PairData(x_atom=atom_node, edge_index_atom=atom_edge_index,
                            #  y_atom=torch.FloatTensor([label]),
                            x_res=res_node, edge_index_res=res_edge_index,
                            y=torch.FloatTensor([label])
                            )  # 使用BCEloss
            # torch.save(data, os.path.join(self.processed_dir, '.pt'))         
            
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data, os.path.join(self.processed_dir, f'{prefix}.pt'))
    
    def len(self):
        number = len(self.data_df)
        return number
    
    def get(self, idx):
        prefix = self.data_df['complex'][idx]
#         print(prefix)
        data = torch.load(os.path.join(self.processed_dir, f'{prefix}.pt'))
        return data


# external test dataset
pos_dir = f'/data5_large/home/xyli/neoantigen/test_model/internal_test/origin_pos/dataset2'
pos_df = pd.read_csv(os.path.join(pos_dir, 'data_info.csv'))
pos_dataset = SageTCRDataset(root=os.path.join(pos_dir, 'language_feature'),
                              data_df=pos_df)

# crossdock
neg_dir1 = f'/data5_large/home/xyli/neoantigen/test_model/internal_test/dock_neg/class1_crossdock_feature/dataset2'
neg_df1 = pd.read_csv(os.path.join(neg_dir1, 'data_info.csv'))
neg_dataset1 = SageTCRDataset(root=os.path.join(neg_dir1, 'language_feature'),
                              data_df=neg_df1)

neg_dir2 = f'/data5_large/home/xyli/neoantigen/test_model/internal_test/dock_neg/class2_crossdock_feature/dataset2'
neg_df2 = pd.read_csv(os.path.join(neg_dir2, 'data_info.csv'))
neg_dataset2 = SageTCRDataset(root=os.path.join(neg_dir2, 'language_feature'),
                              data_df=neg_df2)

# crossalign
neg_dir3 = f'/data5_large/home/xyli/neoantigen/test_model/internal_test/align_neg/class1_crossalign_feature/dataset2'
neg_df3 = pd.read_csv(os.path.join(neg_dir3, 'data_info.csv'))
neg_dataset3 = SageTCRDataset(root=os.path.join(neg_dir3, 'language_feature'),
                              data_df=neg_df3)
neg_dir4 = f'/data5_large/home/xyli/neoantigen/test_model/internal_test/align_neg/class2_crossalign_feature/dataset2'
neg_df4 = pd.read_csv(os.path.join(neg_dir4, 'data_info.csv'))
neg_dataset4 = SageTCRDataset(root=os.path.join(neg_dir4, 'language_feature'),
                              data_df=neg_df4)

neg_dataset = neg_dataset1 + neg_dataset2 + neg_dataset3 + neg_dataset4
dataset = pos_dataset + neg_dataset
pos_num = len(pos_dataset)
neg_num = len(neg_dataset)
# 打包成dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, follow_batch=['x_atom', 'x_res'])

from model.SageTCR import *

ablation = 0
# 1. 初始化模型    
# embedding部分
Embedding_layer = embedding_layer(
                      atom_feature_dim=384,
                      res_feature_dim=446,
                      d_p=512)
# exchange部分
Exchange_layer = res_atom_exchange(feature_dim=512, n_heads=8, dropout=0.3)
# GNN部分
atom_gnn = GNN_layer(feature_dim=512, d_p=256)
atom_lstm = LSTMLayer(embedding_dim=256, hidden_dim=64, num_layers=1, dropout=0.3)  # bidirection 
atom_self_attn = SelfAttention(feature_dim=128, n_heads=4, dropout=0.3)
res_gnn = GNN_layer(feature_dim=512, d_p=256)
res_lstm = LSTMLayer(embedding_dim=256, hidden_dim=64, num_layers=1, dropout=0.3)  # bidirection 
res_self_attn = SelfAttention(feature_dim=128, n_heads=4, dropout=0.3)

Atom_SubForward = SubForward(gnn_layer=atom_gnn, lstm_layer=atom_lstm, self_attention_layer=atom_self_attn)
Res_SubForward = SubForward(gnn_layer=res_gnn, lstm_layer=res_lstm, self_attention_layer=res_self_attn)

if ablation == 0:
    model = SageTCR(embedding_layer=Embedding_layer,
                exchange_layer=Exchange_layer,
                atom_subforward=Atom_SubForward,
                res_subforward=Res_SubForward,
                embedding_dim=256)


param_file = 'params/rotate_perturbation.pt'
model.load_state_dict(torch.load(os.path.join(param_dir,'epoch%d_statedict.pth'%epoch),
                      map_location=torch.device('cpu')))
np.set_printoptions(precision=None, suppress=True)
model.eval()
model.to(device)
pred_label = torch.empty(0,1).to(device)
true_label = torch.empty(0,1).to(device)
scores = torch.empty(0,1).to(device)

with torch.no_grad(): # 表示没有梯度，只测试不调优
    test_loss = 0
    test_right_count = 0
    test_recall = 0

    for data in dataloader:
        data = data.to(device)
        label = data.y.unsqueeze(dim=-1)
        atom_node = data.x_atom
        atom_edge = data.edge_index_atom
        atom_batch = data.x_atom_batch

        res_node = data.x_res
        res_edge = data.edge_index_res
        res_batch = data.x_res_batch
        output, atom_attm, res_attn = model(atom_node, atom_edge, res_node, res_edge, atom_batch, res_batch)

        # 指标（BCE版本）
        output = output.detach()
        batch_score = F.sigmoid(output)
        scores = torch.concat([scores, batch_score], dim=0)
        batch_pred_label = torch.round(F.sigmoid(output))  # 这个batch的预测标签
        pred_label = torch.concat([pred_label, batch_pred_label], dim=0)  # 记录所有样本的预测标签
        true_label = torch.concat([true_label, label.detach()], dim=0)

print(pred_label)
