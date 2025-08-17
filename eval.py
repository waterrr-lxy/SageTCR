from model import *
import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, GraphNorm, global_mean_pool
from torch_geometric.data import Dataset, DataLoader, Data

# from torch.utils.data import Dataset, DataLoader
# from torch_geometric.data import Dataset, DataLoader, Data

import numpy as np
import os
from scipy.sparse import coo_matrix
from data import SageTCRDataset

print('cuda可用性：', torch.cuda.is_available())

gab_freeze_layer = SAGEConv(56, 16, aggr='mean')
grb_freeze_layer = SAGEConv(50, 16, aggr='mean')
Exchange_layer = res_atom_exchange(16,16)

# gnn部分
atom_gnn = GNN_layer(feature=16)
res_gnn = GNN_layer(feature=16)

# feed_forward部分
atom_forward = node_feed_forward(feature=16, d_p=64)
res_forward = node_feed_forward(feature=16, d_p=64)
Atom_SubForward = SubForward(atom_gnn, atom_forward)
Res_SubForward = SubForward(res_gnn, res_forward)

model = SageTCR(gab_freeze_layer=gab_freeze_layer,
                grb_freeze_layer=grb_freeze_layer,
                exchange_layer=Exchange_layer,
                atom_subforward=Atom_SubForward,
                res_subforward=Res_SubForward)

print(model)


# 读入数据
dataset_dir = '/data4_large1/home_data/xyli/neoantigen/ref_model/SAGERank/database/casestudy_peptidescan/gab_nor_dataset'
npz_dir = '/data4_large1/home_data/xyli/neoantigen/ref_model/SAGERank/database/casestudy_peptidescan/gab_nor_npz_complex'
label = 0

gab_dataset = SageTCRDataset(root=dataset_dir,
                             npz_dir=npz_dir,
                             label=label
                            )

dataset_dir = '/data4_large1/home_data/xyli/neoantigen/ref_model/SAGERank/database/casestudy_peptidescan/grb_nor_dataset'
npz_dir = '/data4_large1/home_data/xyli/neoantigen/ref_model/SAGERank/database/casestudy_peptidescan/grb_nor_npz'
label = 0
grb_dataset = SageTCRDataset(root=dataset_dir,
                             npz_dir=npz_dir,
                             label=label
                            )

gab_loader = DataLoader(gab_dataset, batch_size=8, shuffle=False)
grb_loader = DataLoader(grb_dataset, batch_size=8, shuffle=False)

device = torch.device('cuda:2')
model.to(device)
model.load_state_dict(torch.load('/data4_large1/home_data/xyli/neoantigen/ref_model/my_model/epoch50_statedict.pth'))
np.set_printoptions(precision=4, suppress=True)
with torch.no_grad(): # 表示没有梯度，只测试不调优
    model.eval()
    # right_count = 0
    # dataset_loss = 0
    
    for gab_data, grb_data in zip(gab_loader, grb_loader):
        gab_data.to(device)
        grb_data.to(device)
        gab_data = gab_data.sort(sort_by_row=False)
        grb_data = grb_data.sort(sort_by_row=False)
        label = gab_data.y

        atom_node = gab_data.x
        atom_edge = gab_data.edge_index
        res_node = grb_data.x
        res_edge = grb_data.edge_index
        atom_batch = gab_data.batch
        res_batch = grb_data.batch

        output = model(atom_node, atom_edge, res_node, res_edge, atom_batch, res_batch)
        print(F.softmax(output).detach().cpu().numpy())