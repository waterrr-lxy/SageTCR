from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch
from scipy.sparse import coo_matrix
import shutil
import os
import pandas as pd
import numpy as np
from typing import Any

'''定义数据集'''
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

# ----------------------main-------------------------
npz_dir = f'/data4_large1/home_data/xyli/neoantigen/github_opensource/example/lm_npz'
dataset_dir = f'/data4_large1/home_data/xyli/neoantigen/github_opensource/example/dataset'
label = 'unknown'
os.system(f'mkdir {dataset_dir}')

npz_ls = os.listdir(npz_dir)
npz_ls.sort()
df = pd.DataFrame(columns=['complex', 'npz_path', 'label'])
for filename in npz_ls:
    prefix = filename[:-4]
    npz_path = os.path.join(npz_dir, filename)
    df.loc[len(df)] = [prefix, npz_path, label]
df.to_csv(os.path.join(dataset_dir, 'data_info.csv'), index=None)

dataset_root = os.path.join(dataset_dir, 'language_feature')
try:
    shutil.rmtree(dataset_root)
except:
    pass
dataset = SageTCRDataset(root=dataset_root,
                         data_df=df)