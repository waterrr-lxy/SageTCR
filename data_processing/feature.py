from rdkit.Chem.rdmolfiles import MolFromPDBFile
from rdkit import Chem
import numpy as np
import torch
from torch import nn
# from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline  # chemberta
# from transformers import EsmTokenizer, EsmForMaskedLM, pipeline  # saprot

# 用pdb2sql似乎可以快速获取CA coord?
from pdb2sql import pdb2sql
from scipy.spatial import distance_matrix
import os
import sys
sys.path.append(
    "/data4_large1/home_data/xyli/pretrained/SaProt"
)
from utils.foldseek_util import get_struc_seq

def get_CA_adj(pdb1, pdb2):
    db1 = pdb2sql(pdb1)
    ca_coords1 = db1.get('x,y,z', name=['CA'])
    db2 = pdb2sql(pdb2)
    ca_coords2 = db2.get('x,y,z', name=['CA'])
    CA_adj = (distance_matrix(ca_coords1, ca_coords2) <=8).astype(int)

    return CA_adj


class Atom_feature:
    '''
    获取残基的chemberta特征：get_chemberta_feature
    '''
    
    def __init__(self, pdb_file, feature_extractor):
        self.pdb = pdb_file
        self.mol = MolFromPDBFile(pdb_file, sanitize=False)
        # self.mol_noh = Chem.RemoveHs(self.mol)  # 去掉H原子
        self.feature_extractor = feature_extractor  # chemberta的pipeline
        self.aa_smile_dict = {
                            'ALA': 'C[C@H](N)C(=O)',
                            'CYS': 'N[C@@H](CS)C(=O)',
                            'ASP': 'N[C@@H](CC(=O)O)C(=O)',
                            'GLU': 'N[C@@H](CCC(=O)O)C(=O)',
                            'PHE': 'N[C@@H](Cc1ccccc1)C(=O)',
                            'GLY': 'NCC(=O)',
                            'HIS': 'N[C@@H](Cc1c[nH]cn1)C(=O)',
                            'ILE': 'CC[C@H](C)[C@H](N)C(=O)',
                            'LYS': 'NCCCC[C@H](N)C(=O)',
                            'LEU': 'CC(C)C[C@H](N)C(=O)',
                            'MET': 'CSCC[C@H](N)C(=O)',
                            'ASN': 'NC(=O)C[C@H](N)C(=O)',
                            'PRO': 'N1CCC[C@@H]1C(=O)',
                            'GLN': 'NC(=O)CC[C@H](N)C(=O)',
                            'ARG': 'N=C(N)NCCC[C@H](N)C(=O)',
                            'SER': 'N[C@@H](CO)C(=O)',
                            'THR': 'C[C@@H](O)[C@H](N)C(=O)',
                            'VAL': 'CC(C)[C@H](N)C(=O)',
                            'TRP': 'N[C@@H](Cc1c[nH]c2ccccc12)C(=O)',
                            'TYR': 'N[C@@H](Cc1ccc(O)cc1)C(=O)',
                            }


    # 这个类里面的方法    
    # 将残基看作一个小分子，获取其chemberta特征
    def get_chemberta_feature(self, resname, C_terminal=False, NCAA=False, NCAA_smile=None):
        '''
        pipeline: transformers.pipeline  "feature-extraction'  注意去掉decoder层
        resname: 氨基酸的三字母
        C_terminal: 氨基酸是否是C端，如果是C端需要补上末端O，默认False
        NCAA：bool，是否是PTM或非天然氨基酸，默认False
        NCAA_smile：如果NCAA=True，这个氨基酸的smile需要自己提供（如果是C端请在提供NCAA_smile的时候就加上末端O）

        【Usage example】:

        from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline
        import os

        model = AutoModelWithLMHead.from_pretrained("/data4_large1/home_data/xyli/pretrained/ChemBERTa/ChemBERTa-77M-MTR/")
        tokenizer = AutoTokenizer.from_pretrained("/data4_large1/home_data/xyli/pretrained/ChemBERTa/ChemBERTa-77M-MTR/")

        # 去掉decoder层
        model.lm_head = nn.Sequential()
        feature_extraction = pipeline('feature-extraction', model=model, tokenizer=tokenizer)  # out_dim:384

        atom_feature.get_chemberta_feature(feature_extraction, 'MET')

        '''
        pipeline = self.feature_extractor
        if NCAA:  # 如果是非天然氨基酸，需要提供NCAA的smiles
            aa_smile = NCAA_smile
        else:    # 如果是标准氨基酸，从字典里面读取信息
            aa_smile = self.aa_smile_dict[resname]
            if C_terminal:
                aa_smile = aa_smile + 'O'  # 如果是C端，补上C端的O

        chemberta_feature = np.array(pipeline(aa_smile)).squeeze()  
        # [CLS, ....tokens, SEP] 也就是说feature维度比smile_tokens多出两维
        
        return chemberta_feature


    # 获取残基-原子信息字典
    def get_resAtom_info_dict(self):

        '''
        返回值的字典的键值对be like:
        TYR123:
        [12,  # 表示TYR123有12个非H原子
        [atom1_feature, atom2_feature, ..., atom12_feature]]
        '''
        resAtom_info_dict = {}

        # for atom in self.mol_noh.GetAtoms():
        for atom in self.mol.GetAtoms():

            resinfo = atom.GetPDBResidueInfo()
            res_label = resinfo.GetResidueName() + str(resinfo.GetResidueNumber())  # MET123
            if res_label not in resAtom_info_dict :
                resAtom_info_dict[res_label] = self.get_chemberta_feature(res_label[:3])[1:-1]
                # 去掉一头一尾，头是[CLS]，尾是[SEP]
        
        return resAtom_info_dict
    
    # 获取节点邻接矩阵
    def get_atom_data(self):
        '''
        return: atom节点特征矩阵，atom邻接矩阵
        '''
        # 节点特征的矩阵
        resAtom_info_dict = self.get_resAtom_info_dict()
        feature_ls = [feature for reslabel, feature in resAtom_info_dict.items()]
        feature_matrix = np.concatenate(feature_ls, axis=0)  # 拼成atom节点特征矩阵
        atom_adj = np.zeros((feature_matrix.shape[0], feature_matrix.shape[0]))  # 初始化一个全0的atom邻接矩阵
        
        db = pdb2sql(self.pdb)
        ca_coords = db.get('x,y,z', name=['CA'])  # 获取所有CA的坐标
        residue_adj = (distance_matrix(ca_coords, ca_coords) <=8).astype(int)
        # 获取残基水平的邻接矩阵：如果CA之间的距离小于8A，则相连


        # 将残基水平的相连扩展为原子水平的相连（全连接）
        idx_ls = [feature.shape[0] for reslabel, feature in resAtom_info_dict.items()]
        # 每个残基对应的atom_feature的长度，即每个残基对应多少个节点

        for pos in np.argwhere(residue_adj):  # np.argwhere可以获取非0元素的索引
            res1, res2 = pos  # 相互有边的残基的序号

            # res1的占位的区域
            start1 = sum(idx_ls[0:res1])
            end1 = sum(idx_ls[0:res1+1])

            # res2的占位的区域
            start2 = sum(idx_ls[0:res2])
            end2 = sum(idx_ls[0:res2+1])

            atom_adj[start1:end1, start2:end2] = 1

        return feature_matrix, atom_adj
     

class Res_feature:

    def __init__(self, fulllength_pdb, interface_pdb, feature_extractor):

        self.fulllength_pdb = fulllength_pdb  # 完整蛋白的PDB文件
        self.interface_pdb = interface_pdb   # 界面蛋白的PDB文件
        self.feature_extractor = feature_extractor  # saprot的pipeline
    

    # 获取reslabel的列表
    def get_reslabel_ls(self, pdb_file, chain_id):

        '''
        返回resname+resid的列表 e.g.  [TYR123, ARG124, TRP125 ... ]
        '''
        mol = MolFromPDBFile(pdb_file, sanitize=False)
        # mol_noh = Chem.RemoveHs(mol)  # 去掉H原子

        reslabel_ls = []
        # for atom in mol_noh.GetAtoms():
        for atom in mol.GetAtoms():
            resinfo = atom.GetPDBResidueInfo()
            if resinfo.GetChainId() != chain_id:  # 如果不是需要的链，continue
                 continue
            
            res_label = resinfo.GetResidueName() + str(resinfo.GetResidueNumber())  # MET123
            if res_label not in reslabel_ls:
                reslabel_ls.append(res_label)
                # 去掉一头一尾，头是[CLS]，尾是[SEP]
        
        return reslabel_ls
    
    # 获取全长蛋白的foldseek 序列化的结构
    def get_strucseq(self, chain_id):
        '''
        foldseek_path: foldseek binary file的路径
        fulllength_pdb: 完整蛋白的pdb文件（不是界面）
        chain_id:需要提取特征的是哪一条链
        '''
        foldseek_path = '/data4_large1/home_data/xyli/pretrained/SaProt/bin/foldseek'
        parsed_seqs = get_struc_seq(foldseek_path,
                                    self.fulllength_pdb,
                                    chains=None,
                                    plddt_mask=False)
        seq, foldseek_seq, combined_seq = parsed_seqs[chain_id]
        print(f"seq: {seq}")
        print(f"foldseek_seq: {foldseek_seq}")
        print(f"combined_seq: {combined_seq}")

        return combined_seq
    
    # 获取界面的saprot特征
    def get_res_data(self, chain_id):

        # 获取全长蛋白的saprot feature
        pipeline = self.feature_extractor
        fl_strucseq = self.get_strucseq(chain_id)  # 获取全长蛋白的stru-seq
        fl_saprot_feature = (np.array(pipeline(fl_strucseq)).squeeze())[1:-1]
        # 去掉一头的<cls>和一尾的<eos>

        # 分别获取全长和界面的reslabel_ls，寻找界面在全长里面的idx，map过去
        fl_reslabel_ls = self.get_reslabel_ls(self.fulllength_pdb, chain_id)
        if_reslabel_ls = self.get_reslabel_ls(self.interface_pdb, chain_id)
        map_idx = [fl_reslabel_ls.index(reslabel) for reslabel in if_reslabel_ls]
        feature_matrix = fl_saprot_feature[map_idx]

        # 获取邻接矩阵
        # db = pdb2sql(self.interface_pdb)
        # ca_coords = db.get('x,y,z', name=['CA'])  # 获取所有CA的坐标
        # residue_adj = (distance_matrix(ca_coords, ca_coords) <=8).astype(int)
        residue_adj = get_CA_adj(self.interface_pdb, self.interface_pdb)
        # 获取残基水平的邻接矩阵：如果CA之间的距离小于8A，则相连

        return feature_matrix, residue_adj
    

class Complex_feature():

    def __init__(self, 
                 rec_fulllength_pdb,
                 rec_interface_pdb,
                 lig_fulllength_pdb,
                 lig_interface_pdb,
                 atom_feature_extractor,
                 res_feature_extractor, ) -> None:
        
        self.rec_fulllength_pdb = rec_fulllength_pdb
        self.rec_interface_pdb = rec_interface_pdb
        self.lig_fulllength_pdb = lig_fulllength_pdb
        self.lig_interface_pdb = lig_interface_pdb
        

        # 特征提取器
        self.atom_feature_extractor = atom_feature_extractor
        self.res_feature_extractor = res_feature_extractor

    def get_atom_data(self):
        rec_atom_feature = Atom_feature(self.rec_interface_pdb,
                                        self.atom_feature_extractor)
        lig_atom_feature = Atom_feature(self.lig_interface_pdb,
                                        self.atom_feature_extractor)
        
        rec_node, rec_adj = rec_atom_feature.get_atom_data()
        lig_node, lig_adj = lig_atom_feature.get_atom_data()
        complex_node = np.concatenate([rec_node, lig_node], axis=0)

        complex_adj  = np.zeros((complex_node.shape[0], complex_node.shape[0]))

        # crosslink部分(右上角那块)，左下角需要转置   shape: rec_num, lig_num
        atom_crosslink = np.zeros((rec_node.shape[0], lig_node.shape[0]))
        rec_resatom_info_dict = rec_atom_feature.get_resAtom_info_dict()
        lig_resatom_info_dict = lig_atom_feature.get_resAtom_info_dict()

        # rec_db = pdb2sql(self.rec_interface_pdb)
        # rec_ca_coords = rec_db.get('x,y,z', name=['CA'])
        # lig_db = pdb2sql(self.lig_interface_pdb)
        # lig_ca_coords = lig_db.get('x,y,z', name=['CA'])
        # res_crosslink = (distance_matrix(rec_ca_coords, lig_ca_coords) <=8).astype(int)
        res_crosslink = get_CA_adj(self.rec_interface_pdb,
                                   self.lig_interface_pdb,)

        rec_idx_ls = [feature.shape[0] for reslabel,feature in rec_resatom_info_dict.items()]
        lig_idx_ls = [feature.shape[0] for reslabel,feature in lig_resatom_info_dict.items()]

        for pos in np.argwhere(res_crosslink):
            res1, res2 = pos

            # rec占位的区域
            start1 = sum(rec_idx_ls[0:res1])
            end1 = sum(rec_idx_ls[0:res1+1])

            # lig占位的区域
            start2 = sum(lig_idx_ls[0:res2])
            end2 = sum(lig_idx_ls[0:res2+1])

            atom_crosslink[start1:end1, start2:end2] = 1

        complex_adj[:rec_node.shape[0], :rec_node.shape[0]] = rec_adj
        complex_adj[rec_node.shape[0]:, rec_node.shape[0]:] = lig_adj
        complex_adj[:rec_node.shape[0], rec_node.shape[0]:] = atom_crosslink
        complex_adj[rec_node.shape[0]:, :rec_node.shape[0]] = atom_crosslink.T

        return complex_node, complex_adj


    def get_res_data(self):
        rec_res_feature = Res_feature(self.rec_fulllength_pdb,
                                      self.rec_interface_pdb,
                                      self.res_feature_extractor)
        lig_res_feature = Res_feature(self.lig_fulllength_pdb,
                                      self.lig_interface_pdb,
                                      self.res_feature_extractor)
        
        rec_node, rec_adj = rec_res_feature.get_res_data(chain_id='D')
        lig_node, lig_adj = lig_res_feature.get_res_data(chain_id='C')

        complex_node = np.concatenate([rec_node, lig_node], axis=0)
        complex_adj = np.zeros((complex_node.shape[0], complex_node.shape[0]))
        res_crosslink = get_CA_adj(self.rec_interface_pdb, self.lig_interface_pdb)
        complex_adj[:rec_node.shape[0], :rec_node.shape[0]] = rec_adj
        complex_adj[rec_node.shape[0]:, rec_node.shape[0]:] = lig_adj
        complex_adj[:rec_node.shape[0], rec_node.shape[0]:] = res_crosslink
        complex_adj[rec_node.shape[0]:, :rec_node.shape[0]] = res_crosslink.T

        return complex_node, complex_adj
