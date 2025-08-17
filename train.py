from torch import nn
from torch.nn.utils import clip_grad_norm_  # 梯度裁剪
import torch.nn.functional as F
from torch import randperm
from torch_geometric.nn import SAGEConv, GraphNorm, global_mean_pool, SAGPooling, global_max_pool
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from typing import Any
from scipy.sparse import coo_matrix
from sklearn.metrics import recall_score, precision_score
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import os
import sys
import configparser
# from data.data import SageTCRDataset
import wandb
print('cuda可用性：', torch.cuda.is_available())
from model.SageTCR import *


# 定义数据集
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
        
        super(SageTCRDataset, self).__init__(root, transform, pre_transform, pre_filter)
     
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):

        return [f'{prefix}.pt' for prefix in self.complex_ls]
    
    def download(self):
        pass

    def process(self):

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


# 0.读取config文件
config = configparser.ConfigParser()
config_file = sys.argv[1]
config.read(config_file)   # 读取config文件

batch_size = eval(config.get('dataset', 'batch_size'))  # 4
pos_type = config.get('dataset', 'pos_type')  # rotate/selfdock
epoch = eval(config.get('train', 'epoch'))  # 200
lr = eval(config.get('train', 'lr'))  # 1e-3
weight_decay = eval(config.get('train', 'weight_decay'))
accumulate_steps = eval(config.get('train', 'accumulate_steps'))  # 8
param_log = config.get('train', 'param_dir')
cuda_id = config.get('train', 'cuda')
ablation = eval(config.get('train', 'ablation'))  # 0
wandb_name = config.get('wandb', 'name')

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
elif ablation == 1:     
    model = Atom_only(embedding_layer=Embedding_layer,
                    exchange_layer=Exchange_layer,
                    atom_subforward=Atom_SubForward,
                    res_subforward=Res_SubForward,
                    embedding_dim=128)
elif ablation == 2:
    model = Res_only(embedding_layer=Embedding_layer,
                    exchange_layer=Exchange_layer,
                    atom_subforward=Atom_SubForward,
                    res_subforward=Res_SubForward,
                    embedding_dim=128)
elif ablation == 3:
    model = No_exchange(embedding_layer=Embedding_layer,
                    exchange_layer=Exchange_layer,
                    atom_subforward=Atom_SubForward,
                    res_subforward=Res_SubForward,
                    embedding_dim=256)


# 2. 载入数据
# man_pos
dataset_dir = f'/data5_large/home/xyli/neoantigen/database/man_pos_data/mhc1_{pos_type}_feature/dataset2'
data_df = pd.read_csv(os.path.join(dataset_dir, 'data_info.csv'))
mhc1_man_pos_dataset = SageTCRDataset(root=os.path.join(dataset_dir, 'language_feature'),
                              data_df=data_df)

# align_neg
dataset_dir = '/data5_large/home/xyli/neoantigen/database/align_neg_data/mhc1_feature/dataset2'
data_df = pd.read_csv(os.path.join(dataset_dir, 'data_info.csv'))
mhc1_align_neg_dataset = SageTCRDataset(root=os.path.join(dataset_dir, 'language_feature'),
                              data_df=data_df)

# crossdock_neg
dataset_dir = '/data5_large/home/xyli/neoantigen/database/dock_neg_data/mhc1_crossdock_feature/dataset2'
data_df = pd.read_csv(os.path.join(dataset_dir, 'data_info.csv'))
mhc1_crossdock_neg_dataset = SageTCRDataset(root=os.path.join(dataset_dir, 'language_feature'),
                              data_df=data_df)

# selfdock_neg
dataset_dir = '/data5_large/home/xyli/neoantigen/database/dock_neg_data/mhc1_selfdock_feature/dataset2'
data_df = pd.read_csv(os.path.join(dataset_dir, 'data_info.csv'))
mhc1_selfdock_neg_dataset = SageTCRDataset(root=os.path.join(dataset_dir, 'language_feature'),
                          data_df=data_df)

# decoydock_neg
dataset_dir = '/data5_large/home/xyli/neoantigen/database/decoygen_neg_data/mhc1_dock_sample/dataset2'
data_df = pd.read_csv(os.path.join(dataset_dir, 'data_info.csv'))
mhc1_decoydock_neg_dataset = SageTCRDataset(root=os.path.join(dataset_dir, 'language_feature'),
                              data_df=data_df)


# decoyalign_neg
dataset_dir = '/data5_large/home/xyli/neoantigen/database/decoygen_neg_data/mhc1_align/dataset2'
data_df = pd.read_csv(os.path.join(dataset_dir, 'data_info.csv'))
mhc1_decoyalign_neg_dataset = SageTCRDataset(root=os.path.join(dataset_dir, 'language_feature'),
                              data_df=data_df)

# mhc2

# man_pos
dataset_dir = f'/data5_large/home/xyli/neoantigen/database/man_pos_data/mhc2_{pos_type}_feature/dataset2'
data_df = pd.read_csv(os.path.join(dataset_dir, 'data_info.csv'))
mhc2_man_pos_dataset = SageTCRDataset(root=os.path.join(dataset_dir, 'language_feature'),
                              data_df=data_df)

# align_neg
dataset_dir = '/data5_large/home/xyli/neoantigen/database/align_neg_data/mhc2_feature/dataset2'
data_df = pd.read_csv(os.path.join(dataset_dir, 'data_info.csv'))
mhc2_align_neg_dataset = SageTCRDataset(root=os.path.join(dataset_dir, 'language_feature'),
                              data_df=data_df)

# crossdock_neg
dataset_dir = '/data5_large/home/xyli/neoantigen/database/dock_neg_data/mhc2_crossdock_feature/dataset2'
data_df = pd.read_csv(os.path.join(dataset_dir, 'data_info.csv'))
mhc2_crossdock_neg_dataset = SageTCRDataset(root=os.path.join(dataset_dir, 'language_feature'),
                              data_df=data_df)

# sefdock_neg
dataset_dir = '/data5_large/home/xyli/neoantigen/database/dock_neg_data/mhc2_selfdock_feature/dataset2'
data_df = pd.read_csv(os.path.join(dataset_dir, 'data_info.csv'))
mhc2_selfdock_neg_dataset = SageTCRDataset(root=os.path.join(dataset_dir, 'language_feature'),
                              data_df=data_df)

# decoydock_neg
dataset_dir = '/data5_large/home/xyli/neoantigen/database/decoygen_neg_data/mhc2_dock_sample/dataset2'
data_df = pd.read_csv(os.path.join(dataset_dir, 'data_info.csv'))
mhc2_decoydock_neg_dataset = SageTCRDataset(root=os.path.join(dataset_dir, 'language_feature'),
                              data_df=data_df)


# decoyalign_neg
dataset_dir = '/data5_large/home/xyli/neoantigen/database/decoygen_neg_data/mhc2_align/dataset2'
data_df = pd.read_csv(os.path.join(dataset_dir, 'data_info.csv'))
mhc2_decoyalign_neg_dataset = SageTCRDataset(root=os.path.join(dataset_dir, 'language_feature'),
                              data_df=data_df)

trn_dataset = mhc1_man_pos_dataset + mhc2_man_pos_dataset + mhc1_align_neg_dataset + mhc2_align_neg_dataset \
+ mhc1_crossdock_neg_dataset + mhc2_crossdock_neg_dataset + mhc1_selfdock_neg_dataset + mhc2_selfdock_neg_dataset \
+ mhc1_decoydock_neg_dataset + mhc2_decoydock_neg_dataset + mhc1_decoyalign_neg_dataset + mhc2_decoyalign_neg_dataset


'''test set'''
# internal dataset
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
val_dataset = pos_dataset + neg_dataset


# 打包成dataloader
trn_dataloader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True, follow_batch=['x_atom', 'x_res'])
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, follow_batch=['x_atom', 'x_res'])


# # 开一个wandb窗口进行记录
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="SageTCR",
#     group='202411',
#     job_type='model8_language_model',
#     name=wandb_name,
#     notes='语言模型',

#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": "Adam",
#     "architecture": "atom_res_exchange",
#     "initial": "kaiming_normal",
#     "dataset": "man_pos + crossalign + crossdock + selfdock",
#     "epochs": 200,
#     }

# )

# 训练

device = torch.device(cuda_id)
model = model.to(device)
model.train()
try:
    os.mkdir(param_log)
except:
    pass
    
# 参数初始化
for name, param in model.named_parameters():
    if 'sagerank' in name:
        param.requires_grad=False
        continue
    if 'weight' in name:
        if param.dim() == 1:
            nn.init.kaiming_normal_(param.unsqueeze(0))  
            # 要2维及以上才能用xavier和kaming初始化
        else:
            nn.init.kaiming_normal_(param)
    elif 'bias' in name:
        nn.init.zeros_(param)
        
# 接着上次训练好的继续训练
# model.load_state_dict(torch.load('/data4_large1/home_data/xyli/neoantigen/ref_model/my_model/param_log/ablation/model10_lm/rotate_pos/epoch9_statedict.pth'))
# for name, param in model.named_parameters():
#     if 'sagerank' in name:
#         param.requires_grad=False

# 优化器和损失函数

optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
loss_fn.to(device)

for i in range(epoch):
    train_loss = 0
    train_right_count = 0
    train_recall = 0
    train_precision = 0
    model.train()
    optimizer.zero_grad()
    for j, data in enumerate(trn_dataloader):
        data = data.to(device)
        label = data.y.unsqueeze(dim=-1)
        atom_node = data.x_atom
        atom_edge_index = data.edge_index_atom
        atom_batch = data.x_atom_batch

        res_node = data.x_res
        res_edge_index = data.edge_index_res
        res_batch = data.x_res_batch
    
        output, atom_attn, res_attn = model(atom_node, atom_edge_index, res_node, res_edge_index, atom_batch, res_batch)
        loss = loss_fn(output, label)
        loss_perstep = loss / accumulate_steps   # 相对于对累加后的梯度取平均
        loss_perstep.backward()
        
        if (j+1) % accumulate_steps == 0: # 梯度累计，每8步进行一次更新
            optimizer.step()
            optimizer.zero_grad()

        # 指标（BCE版本）
        output = output.detach()
        pred_class = torch.round(F.sigmoid(output))
        train_right_count += (pred_class==label).sum()
        train_loss += loss    
        train_recall += recall_score(label.cpu().numpy(), pred_class.cpu().numpy(), average='binary')
        train_precision += precision_score(label.cpu().numpy(), pred_class.cpu().numpy(), average='binary')
        
    print(f'epoch{i+1}:')
    print(f'整个训练集上的loss:{train_loss/len(trn_dataloader)}')
    print(f'整个训练集上的accuracy:{train_right_count/len(trn_dataset)}')
    print(f'整个训练集上的recall:{train_recall/len(trn_dataloader)}')
    print(f'整个训练集上的precision:{train_precision/len(trn_dataloader)}')
    
    # eval
    with torch.no_grad(): # 表示没有梯度，只测试不调优
        test_loss = 0
        test_right_count = 0
        test_recall = 0
        test_precision = 0
        model.eval()
        for data in val_dataloader:
            data = data.to(device)
            label = data.y.unsqueeze(dim=-1)
            atom_node = data.x_atom
            atom_edge_index = data.edge_index_atom
            atom_batch = data.x_atom_batch

            res_node = data.x_res
            res_edge_index = data.edge_index_res
            res_batch = data.x_res_batch

            output, atom_attn, res_attn = model(atom_node, atom_edge_index, res_node, res_edge_index, atom_batch, res_batch)
            print('atom attention weight:', atom_attn)
            print('res attention weight:', res_attn)
            loss = loss_fn(output, label)
            
            # 指标（BCE版本）
            output = output.detach()
            pred_class = torch.round(F.sigmoid(output))
            test_right_count += (pred_class==label).sum()
            test_loss += loss   
            test_recall += recall_score(label.cpu().numpy(), pred_class.cpu().numpy(), average='binary')
            test_precision += precision_score(label.cpu().numpy(), pred_class.cpu().numpy(), average='binary')
    
#         print(f'epoch{i+1}:')
        print(f'整个测试集上的loss:{test_loss/len(val_dataloader)}')
        print(f'整个测试集上的accuracy:{test_right_count/len(val_dataset)}')
        print(f'整个测试集上的recall:{test_recall/len(val_dataloader)}')
        print(f'整个测试集上的precision:{test_precision/len(val_dataloader)}')
        
        
        # wandb.log({
        #             "Train Loss": train_loss/len(trn_dataloader),
        #             "Test Loss": test_loss/len(val_dataloader),
        #             "Train Accuracy": train_right_count/len(trn_dataset),
        #             "Test Accuracy": test_right_count/len(val_dataset),
        #             "Train Recall": train_recall/len(trn_dataloader),
        #             "Test Recall": test_recall/len(val_dataloader),
        #             "Train Precision": train_precision/len(trn_dataloader),
        #             "Test Precision": test_precision/len(val_dataloader),
        #           }, step=i+1)
        
        torch.save(model.state_dict(), os.path.join(param_log, f'epoch{i+1}_statedict.pth')) 

        
# wandb.finish()  