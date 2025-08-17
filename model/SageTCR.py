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

# 最大最小归一化，因为node_embedding和res_embedding的scale不一样，一个是零点几，一个是二十几
def min_max_normaliztion(embedding):
    
    mean = embedding.mean(dim=0, keepdim=True)  # 按列计算均值
    min_val = embedding.min(dim=0, keepdim=True)[0]
    max_val = embedding.max(dim=0, keepdim=True)[0]
    normalized_embedding = (embedding - mean) / (max_val - min_val + 1e-12)

    return normalized_embedding

def standardization(embedding):
    
    mean = embedding.mean(dim=0, keepdim=True)  # 按列计算均值
    std = embedding.std(dim=0, keepdim=True)    # 按列计算标准差
    normalized_embedding = (embedding - mean) / (std + 1e-12)  # 避免除以零

    return normalized_embedding
    
    
class embedding_layer(nn.Module):
    
    def __init__(self, atom_feature_dim, res_feature_dim, d_p):
        super(embedding_layer, self).__init__()
        
        self.atom_fc = nn.Linear(atom_feature_dim, d_p) 
        self.res_fc = nn.Linear(res_feature_dim, d_p)
        
    def forward(self, atom_node, res_node):
#         atom_node = standardization(atom_node)
        atom_node = self.atom_fc(atom_node)
        atom_node = F.relu(atom_node)
        
#         res_node = standardization(res_node)
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

        return atom_node, res_node
    
    
class GNN_layer(nn.Module):
    '''
    3层GNN
    考虑用lstm做aggregation
    '''
    def __init__(self, feature_dim, d_p):
        
        super(GNN_layer, self).__init__()
#         self.layernorm = nn.LayerNorm(normalized_shape=feature_dim)
        self.conv1 = SAGEConv(in_channels=feature_dim, out_channels=d_p, project=True, aggr='mean')
        self.norm1 = GraphNorm(d_p)
        
        self.conv2 = SAGEConv(in_channels=d_p, out_channels=d_p, project=True, aggr='mean')
        self.norm2 = GraphNorm(d_p)
        
        self.conv3 = SAGEConv(in_channels=d_p, out_channels=d_p, project=True, aggr='mean')
        self.norm3 = GraphNorm(d_p)  # 三层似乎会导致过度平滑
        
    def forward(self, x, edge_index, batch):

        output = self.conv1(x, edge_index)
        output = F.relu(output)
        output = self.norm1(output, batch)
        
        output = self.conv2(output, edge_index)
        output = F.relu(output)
        output = self.norm2(output, batch)

        output = self.conv3(output, edge_index)
        output = F.relu(output)
        output = self.norm3(output, batch)
        
        return output
    
# lstm
class LSTMLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout=0.3):
        super(LSTMLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=embedding_dim,  # 输入的序列长度？
                hidden_size=hidden_dim,  # embedding的维度（可以任意设置？一般设置多少？）
                num_layers=num_layers,
                bias=True,
                batch_first=True,
                bidirectional=True,
                dropout=dropout) 
    
    # 注意以后写rnn后面套东西的都要加这个init_hidden存一下隐藏层的值
    # 因为可能会涉及到多方向反向传播会反复调用，不存的话传播一次就被释放掉了，会报错
    def init_hidden(self):
        hidden = torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim), requires_grad=True)  # 维度不重要，会自己更新的
        cell = torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim), requires_grad=True)
        return (hidden, cell)
    
    def forward(self, node, batch):
        '''
        首先to_dense_batch处理GNN传来的东西
        '''
        node_dense, mask = to_dense_batch(node, batch)

        # pad和pack
        lengths = torch.bincount(batch)  # 获得每个data.x的长度
        lengths = lengths.cpu()  # lengths参数必须在cpu上，否则pack_paded_sequences会报错
        # torch.bincount(input, weights=None, minlength=0) → tensor
        # 计算一维整数tensor中每个非负整数出现的频次    
        
        # node_dense本质是一个pading好的batch，可以当成序列数据pack
        node_packed = pack_padded_sequence(input=node_dense, lengths=lengths, batch_first=True, enforce_sorted=False)

        self.hidden = self.init_hidden()  # 要加这一句
        node_output, self.hidden = self.lstm(node_packed) # 过lstm

        # 过完lstm要先把pack的数据pad回Tensor，才能被其他模块处理
        node_output, lengths = pad_packed_sequence(node_output, batch_first=True)
        node = node_output[mask]  # pad好的数据恢复为graphdata.batch打包的形式

        return node, self.hidden
    
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
    def __init__(self, gnn_layer, lstm_layer, self_attention_layer):
        super(SubForward, self).__init__()
        self.gnn = gnn_layer
        self.lstm = lstm_layer
        self.attention = self_attention_layer
        
    def forward(self, node, edge_index, batch):
        
        node = self.gnn(node, edge_index, batch)
        node, hidden = self.lstm(node, batch)
        node, attn_w = self.attention(node, batch)
        output = global_max_pool(node, batch)
        
        return output, attn_w


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
        
        atom_output, atom_attn = self.atom_module(atom_node, atom_edge, atom_batch)
        res_output, res_attn = self.res_module(res_node, res_edge, res_batch)

        x = torch.cat([atom_output, res_output], axis=1)
        x = self.fc(x)
    
        return x, atom_attn, res_attn

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