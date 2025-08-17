'''
2024/11/16: data采用了pairdata的方式，特征采用chemberta化学语言模型和saprot蛋白质语言模型特征
因为维度不匹配，去掉了师兄sagerank的那一层的
'''
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GraphNorm, global_mean_pool, SAGPooling, global_max_pool
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.utils import to_dense_batch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pandas as pd
import numpy as np
import sys
from scipy.sparse import coo_matrix


class embedding_layer(nn.Module):
    
    def __init__(self, atom_feature_dim, res_feature_dim, d_p):
        super(embedding_layer, self).__init__()
        
        self.atom_fc = nn.Linear(atom_feature_dim, d_p) 
        self.res_fc = nn.Linear(res_feature_dim, d_p)
        
    def forward(self, atom_node, res_node):
        atom_node = self.atom_fc(atom_node)
        atom_node = F.relu(atom_node)
        res_node = self.res_fc(res_node)
        res_node = F.relu(res_node)
        
        return atom_node, res_node
    
    
# 其实就是crossattention替代原来的nn.linear交换的方法
class res_atom_exchange(nn.Module):
    
    def __init__(self, feature_dim, n_heads, dropout=0.2):
        super(res_atom_exchange, self).__init__()
        self.feature_dim = feature_dim
        self.atom2res = nn.MultiheadAttention(
                        embed_dim=feature_dim, 
                        num_heads=n_heads, 
                        dropout=dropout,
                        batch_first=True)
        self.res2atom = nn.MultiheadAttention(
                        embed_dim=feature_dim, 
                        num_heads=n_heads, 
                        dropout=dropout,
                        batch_first=True)
        # 补充add & norm
        self.atom_norm = nn.LayerNorm(normalized_shape=feature_dim)
        self.res_norm = nn.LayerNorm(normalized_shape=feature_dim)
        
    def forward(self, atom_node, atom_batch, res_node, res_batch):
        # 先把graph_data转化为dense_data
        atom_node_dense, atom_mask = to_dense_batch(atom_node, atom_batch)
        res_node_dense, res_mask = to_dense_batch(res_node, res_batch)
#         print(atom_node_dense.shape)
#         print(res_node_dense.shape)
        # cross_attention
        atom_node_dense, atom_attn_w = self.res2atom(
                                        query=atom_node_dense,
                                        key=res_node_dense,
                                        value=res_node_dense,
                                        key_padding_mask=(res_mask==False)
                                        )
        res_node_dense, res_attn_w = self.atom2res(
                                query=res_node_dense,
                                key=atom_node_dense,
                                value=atom_node_dense,
                                key_padding_mask=(atom_mask==False)
                                )
        # layernorm
        atom_node_dense = self.atom_norm(atom_node_dense)
        res_node_dense = self.res_norm(res_node_dense)

        # 再回到graph_data的状态并add
        atom_node = atom_node_dense[atom_mask] + atom_node
        res_node = res_node_dense[res_mask] + res_node
#         print(atom_node.shape)
#         print(res_node.shape)

        return atom_node, res_node

class GNN_layer(nn.Module):
    '''
    3层GNN
    考虑用lstm做aggregation
    '''
    def __init__(self, feature_dim, d_p):
        
        super(GNN_layer, self).__init__()
        self.conv1 = SAGEConv(in_channels=feature_dim, out_channels=d_p, project=True, aggr='mean')
        self.sag1 = SAGPooling(d_p, 0.5)
        self.norm1 = GraphNorm(d_p)
        
        self.conv2 = SAGEConv(in_channels=d_p, out_channels=d_p, project=True, aggr='mean')
        self.sag2 = SAGPooling(d_p, 0.5)
        self.norm2 = GraphNorm(d_p)
        
        self.conv3 = SAGEConv(in_channels=d_p, out_channels=d_p, project=True, aggr='mean')
        self.sag3 = SAGPooling(d_p, 0.5)
        self.norm3 = GraphNorm(d_p)
        
    def forward(self, x, edge_index, batch):
#         edge_index = edge_index.sort(sort_by_row=False)
#                                      sorted_edge_index, _ = torch.sort(edge_index, dim=1)
        output = self.conv1(x, edge_index)
        output = F.relu(output)
        output = self.norm1(output, batch)
        pooling = self.sag1(output, edge_index, batch=batch)
        output = pooling[0]
        edge_index = pooling[1]
        batch = pooling[3]
        
        output = self.conv2(output, edge_index)
        output = F.relu(output)
        output = self.norm2(output, batch)
        pooling = self.sag2(output, edge_index, batch=batch)
        output = pooling[0]
        edge_index = pooling[1]
        batch = pooling[3]

        output = self.conv3(output, edge_index)
        output = F.relu(output)
        output = self.norm3(output, batch)
        pooling = self.sag3(output, edge_index, batch=batch)
        output = pooling[0]
        edge_index = pooling[1]
        batch = pooling[3]
        
        return output, edge_index, batch

# self-attention
class SelfAttention(nn.Module):
    def __init__(self, feature_dim, n_heads, dropout=0.2):
        super(SelfAttention, self).__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layernorm = nn.LayerNorm(normalized_shape=feature_dim)

    def forward(self, node, batch):
        node_dense, mask = to_dense_batch(node, batch)
        node_dense, attn_w = self.self_attention(
            query=node_dense,
            key=node_dense,
            value=node_dense,
            key_padding_mask=(mask==False)
        )
        node_dense = self.layernorm(node_dense)
        node = node_dense[mask] + node

        return node, attn_w
    
class SubForward(nn.Module):
    def __init__(self, gnn_layer, feed_forward):
        super(SubForward, self).__init__()
        self.gnn_layer = gnn_layer
        self.feed_forward = feed_forward
        
    def forward(self, node, edge_index, batch):
        node, edge_index, batch = self.gnn_layer(node, edge_index, batch)
        node, _ = self.feed_forward(node, batch)
        output = global_max_pool(node, batch)
        
        return output

# 搭整个网络
class SageTCR(nn.Module):
    
    def __init__(self, embedding_layer, exchange_layer, atom_subforward, res_subforward, embedding_dim):
        
        super(SageTCR, self).__init__()
        
        self.embedding = embedding_layer
        
        # crossattention
        self.exchange = exchange_layer
        
        
        # 输入格式不同，不能直接串到一起
        self.atom_module = atom_subforward   # 之后有条件要换成diff_pool
        self.res_module = res_subforward # 之后有条件要换成diff_pool
        
        # mlp
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1))
        
    def forward(self, atom_node, atom_edge, res_node, res_edge, atom_batch, res_batch):
        
        atom_node, res_node = self.embedding(atom_node, res_node)
        
        atom_node, res_node = self.exchange(atom_node, atom_batch, res_node, res_batch)  # 交换信息
        
        atom_output = self.atom_module(atom_node, atom_edge, atom_batch)
        res_output = self.res_module(res_node, res_edge, res_batch)
        # 这里开始必须分两个pipe，参数不能混了，学习一下transformer里面clone的写法？不对，不能clone，因为atom和res的维度不一样
#         atom_node = gnn_layer1(atom, atom_edge)
#         res_node = gnn_layer2(res, res_edge)
        x = torch.cat([atom_output, res_output], axis=1)
        x = self.fc(x)
    
        return x

class Atom_only(nn.Module):
    
    def __init__(self, embedding_layer, exchange_layer, atom_subforward, res_subforward, embedding_dim):
        
        super(Atom_only, self).__init__()
        
        self.embedding = embedding_layer
        
        # 输入格式不同，不能直接串到一起
        self.atom_module = atom_subforward   # 之后有条件要换成diff_pool
        
        # mlp
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1))
        
    def forward(self, atom_node, atom_edge, res_node, res_edge, atom_batch, res_batch):
        
        atom_node, res_node = self.embedding(atom_node, res_node)
        atom_output = self.atom_module(atom_node, atom_edge, atom_batch)
        x = self.fc(atom_output)
    
        return x    

class Res_only(nn.Module):
    
    def __init__(self, embedding_layer, exchange_layer, atom_subforward, res_subforward, embedding_dim):
        
        super(Res_only, self).__init__()
        
        self.embedding = embedding_layer

        self.res_module = res_subforward # 之后有条件要换成diff_pool
        
        # mlp
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1))
        
    def forward(self, atom_node, atom_edge, res_node, res_edge, atom_batch, res_batch):
        
        atom_node, res_node = self.embedding(atom_node, res_node)
        res_output = self.res_module(res_node, res_edge, res_batch)
        x = self.fc(res_output)
    
        return x

class No_exchange(nn.Module):
    
    def __init__(self, embedding_layer, exchange_layer, atom_subforward, res_subforward, embedding_dim):
        
        super(No_exchange, self).__init__()
        
        self.embedding = embedding_layer
        
        # 输入格式不同，不能直接串到一起
        self.atom_module = atom_subforward   # 之后有条件要换成diff_pool
        self.res_module = res_subforward # 之后有条件要换成diff_pool
        
        # mlp
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1))
        
    def forward(self, atom_node, atom_edge, res_node, res_edge, atom_batch, res_batch):
        
        atom_node, res_node = self.embedding(atom_node, res_node)

        atom_output = self.atom_module(atom_node, atom_edge, atom_batch)
        res_output = self.res_module(res_node, res_edge, res_batch)

        x = torch.cat([atom_output, res_output], axis=1)
        x = self.fc(x)
    
        return x
'''
整个模型的初始化示例：
 
# sagerank部分
gab_freeze_layer = SAGEConv(56, 16, aggr='mean')
grb_freeze_layer = SAGEConv(50, 16, aggr='mean')
gab_freeze_layer.load_state_dict(torch.load('/data4_large1/home_data/xyli/neoantigen/ref_model/my_model/sagerank_conv1_param/atom.pth',
                                            map_location=torch.device('cpu')))
grb_freeze_layer.load_state_dict(torch.load('/data4_large1/home_data/xyli/neoantigen/ref_model/my_model/sagerank_conv1_param/res.pth',
                                            map_location=torch.device('cpu')))
Exchange_layer = res_atom_exchange(feature_dim=16, n_heads=4)

atom_gnn = GNN_layer(feature=16)
atom_self_attn = SelfAttention(embedding_dim=16, hidden_dim=64)
res_gnn = GNN_layer(feature=16)
res_self_attn = SelfAttention(feature_dim=16, n_heads=4)

Atom_SubForward = SubForward(gnn_layer=atom_gnn, feed_forward=atom_self_attn)
Res_SubForward = SubForward(gnn_layer=res_gnn, feed_forward=res_self_attn)

model = SageTCR(gab_freeze_layer=gab_freeze_layer,
                grb_freeze_layer=grb_freeze_layer,
                exchange_layer=Exchange_layer,
                atom_subforward=Atom_SubForward,
                res_subforward=Res_SubForward,
                embedding_dim=32)
'''
