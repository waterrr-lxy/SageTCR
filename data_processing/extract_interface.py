# 我直接用pymol做吧
import pymol
from pymol import cmd
import os
import pandas as pd

def extract_interface(pdb_path, lig_dir, rec_dir, lig_chain='C', rec_chain='D', cut_off=10, save_name=None):
    '''
    pdb_path: 需要提取界面的pdb文件的路径
    lig_dir: 提取出来的ligand interface.pdb保存到哪里
    rec_dir: 提取出来的receptor interface.pdb保存到哪里
    lig_chain: pdb文件中ligand的chain ID，默认为C
    rec:chain: pdb文件中receptor的chain ID，默认为D
    cut_off: 两个氨基酸之间的最小距离在多少里内定义为界面，默认为5
    save_name:指定保存interface.pdb的名称（注意应带有.pdb后缀），默认为None，即使用原始pdb文件同样的名字
    '''
    
    # 加载文件
    pdb_name = os.path.split(pdb_path)[1]
    cmd.set('pdb_retain_ids', 0)
    cmd.set('retain_order', 1)
    cmd.load(pdb_path, pdb_name)
    
    # 选择界面
    cmd.select('rec_inter', f'byres chain {rec_chain} within {cut_off} of chain {lig_chain}')
    cmd.select('lig_inter', f'byres chain {lig_chain} within {cut_off} of chain {rec_chain}')

    # 如果有指定的save_name
    if bool(save_name):
        pdb_name = save_name
    # 保存receptor和ligand的interface.pdb
    rec_file = os.path.join(rec_dir, pdb_name)
    lig_file = os.path.join(lig_dir, pdb_name)
    cmd.save(rec_file, selection='rec_inter')
    cmd.save(lig_file, selection='lig_inter')
    cmd.reinitialize()


# sample_info = pd.read_csv('/data4_large1/home_data/xyli/neoantigen/ref_model/SAGERank/database/dock_neg_data/mhc2_selfdock_irmsd_top_v2.csv')
complex_dir = '/data4_large1/home_data/xyli/neoantigen/github_opensource/example/complex_pdb'
rec_dir = '/data4_large1/home_data/xyli/neoantigen/github_opensource/example/rec'
lig_dir = '/data4_large1/home_data/xyli/neoantigen/github_opensource/example/lig'
os.makedirs(rec_dir, exist_ok=True)
os.makedirs(lig_dir, exist_ok=True)


for filename in os.listdir(complex_dir):
    complex_pdb = os.path.join(complex_dir, filename)
    extract_interface(complex_pdb, lig_dir, rec_dir, save_name=filename)

