import pymol
from pymol import cmd
from pymol import stored
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from igmat.igmat import annotate
hmmerpath = '/data2/data_home/xyli/anaconda3/envs/sagerank/bin'
# from MDAnalysis.analysis import *
import numpy as np
from math import sqrt
from Bio import AlignIO
from Bio import PDB
from Bio.PDB import PDBParser

'''
一般
'''

# 从PDB文件中提取序列
def extract_seq(pdb_file):
    seq_dict = {}
    parser = PDB.PDBParser()
    structure_name = pdb_file.replace('.pdb','')
    tcr_structure = parser.get_structure(structure_name, pdb_file)
# 算了，还是先预处理文件吧，让所有TCRA都编号为A，让TCRB都编号为B
    ppb = PDB.PPBuilder()
    chain_id_ascii = ord('A')  
    for pp in ppb.build_peptides(tcr_structure):
        seq = pp.get_sequence()
        seq_dict[chr(chain_id_ascii)] = str(seq)
        chain_id_ascii += 1
    
    return seq_dict

'''
TCR的CDR注释模块
'''
# 根据单链序列标注CDR区
def find_cdr(sequence):
    '''
    输入：TCR一条链的完整序列  e.g. 'GVTQTPKFQ......TVTED'
    返回：三个CDR区的编号（从1开始，不是按pdb里面的）构成的列表 e.g.
    [[24, 25, 26, 27, 28],
    [46, 47, 48, 49, 50, 51],
    [89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102]]
    '''
    cdr_resid = []
    
    resultList = annotate(sequence, 'IMGT', hmmerpath=hmmerpath)
    if not resultList:
        raise Exception('No result found')
    
    for result in resultList:
        for feature in result.annotations():
            print('Annotation {type}: {start}-{stop}'.format(
                 type=feature['type'],
                 start=feature['start'],
                 stop=feature['stop']
                ))
            if 'CDR' in feature['type']:
                cdr_resid.append(list(range(feature['start'], feature['stop']+1)))
                
    return cdr_resid

def extract_cdr_seq(seq_dict):
    '''
    seq_dict: 用extract_seq从tcr_pdb提取出来的序列的字典
    返回：6段CDR序列构成的列表  e.g. ['DRGSQS', 'IYSNGD', 'AVTTDSWGKLQ', 'MNHEY', 'SVGAGI', 'ASRPGLAGGRPEQY']
    '''

    # 先提取序列
    # seq_dict = extract_seq(input_pdb)
    # print('------------------Extracting sequences from pdbfile-------------')
    # print(seq_dict)
    full_seq = ''
    for seq_segment in list(seq_dict.values()):
        full_seq += str(seq_segment)

    # 因为有的TCR seq中间有断开或缺少的氨基酸，导致bio会提出三条链，不完整的链无法被igmat识别注释，导致报错
    # 所以就不强行区分TCRA和TCRB了，直接把整段序列一分为二，反而不影响寻找CDR区
    full_len = len(full_seq)
    first_half = full_seq[:full_len//2]
    second_half = full_seq[full_len//2:]

    cdr_seq_ls = []
    print('----------extract CDRs for TCRA-------------')
    for cdr in find_cdr(first_half):
        cdr_seq_ls.append(first_half[cdr[0]:cdr[-1]+1])

    print('----------extract CDRs for TCRB-------------')
    for cdr in find_cdr(second_half):
        cdr_seq_ls.append(second_half[cdr[0]:cdr[-1]+1])

    return cdr_seq_ls
# class TCR(self):
# ```
#     用一个PDB来初始化TCR这个类，它需要有这些功能：
#     1）根据PDB文件提取序列
#     2）对于从序列中提取CDR区
#     3 从完整的TCR中提取出CDR区部分的pdb形成对接的表面
#     4.对于这部分提出的CDR.pdb，找出质心，进行坐标变化（平移？），使质心的坐标为0,0,0
# ```
#     def __init__(self, tcr_pdb):
#     self.pdb = tcr_pdb

#     # 从PDB文件中提取TCR序列
#     def extract_seq(self.pdb):
#         '''
#         从PDB提取序列，不需要每条链分开编号，只需要有TER区分是不同的链就行了
#         返回一个字典,长下面这样
#         {'A': 'GSHSMYLENGKETLQR', 'B': 'LLFGYPVYV'}
#         '''
#         seq_dict = {}
#         parser = PDB.PDBParser()
#         structure_name = tcr_file.replace('.pdb','')
#         tcr_structure = parser.get_structure(structure_name, tcr_file)
#     # 不需要分开编号，只要有TER能够区分两条链就可以了
#         ppb = PDB.PPBuilder()
#         chain_id_ascii = ord('A')
#         for pp in ppb.build_peptides(tcr_structure):
#             seq = pp.get_sequence()
#             seq_dict[chr(chain_id_ascii)] = str(seq)
#             chain_id_ascii += 1
        
#         return seq_dict


#     # 根据单链序列标注CDR区
#     def find_tcr(sequence):
#         cdr_resid = []
        
#         resultList = annotate(sequence, 'IMGT', hmmerpath=hmmerpath)
#         if not resultList:
#             raise Exception('No result found')

#         for result in resultList:
#             for feature in result.annotations():
#                 if 'CDR' in feature['type']:
#                     cdr_resid += list(range(feature['start'], feature['stop']+1))
#         return cdr_resid

'''
计算ZDOCK complex与正样本irmsd部分所需函数
'''
# pymol部分
# 包成函数:修改TCR编号从1开始连续编号
def pymol_renum(selection):  # e.g. 'pos and chain D'
    # 获取chain D原始的原子编号
    stored.pairs = []
    cmd.iterate("%s and n. ca" % selection, "stored.pairs.append((resi, chain))")

    # 遍历，逐一修改resid
    renum = 1
    for i, chain_id in stored.pairs:
        cmd.alter('%s and chain %s and resi %s' % (selection, chain_id, i), 'resi=%d' % renum)
        renum += 1

    # 打印新的resi，看看是否修改成功
    print('Successfully renumber the resid from 1 !')
    # cmd.alter('%s and n. ca' % selection, 'print(resi)')

# 把前面的步骤包成一个函数
# 首先定义一个函数找出seq中所有的-
def find_char(string, char):
    indexs = []
    for index, c in enumerate(string):
        if c == char:
            indexs.append(index)
    return indexs

# 从interface pdb文件中获取resid list
def get_resid(pdb_path):
    p = PDBParser()
    s1 = p.get_structure('s1', pdb_path)
    resid_ls = []
    residue = s1.get_residues()
    for res in residue:
        resid_ls.append(res.get_id()[1])

    return resid_ls

# 调整resid_ls，使之能够map到带'-'的seq上面
# 基本思想是依次遍历每个'-'的index，如果resid >= '-'的index的都+1，相当于整体向后移动一位，留出'-'的位置
def map2seq(resid_ls, unmatch_idx):  # 【注意：列表和字典是可变的类型，全局的列表和字典传到函数里之后，也会被改掉！】
    new_ls = resid_ls.copy()   # 不能写new_ls=resid_ls 这样写相当于两个变量共享内存地址，只要改一个就会改掉另一个，必须通过拷贝
    for char_position in unmatch_idx:
        for i, resid in enumerate(new_ls):
            if resid-1 >= char_position:  # 因为pdb里面resid编号是从1开始的，但是'-'在seq里面的index是从0开始的，所以要减1
                sub_ls = new_ls[i:]  # map不支持[:]索引切片，只能把要修改的部分取一个子列表出来
                sub_ls = list(map(lambda x:x+1, sub_ls))
                new_ls[i:] = sub_ls
                break
    
    # 把从1开始的resid改为从0开始的index，
    new_ls = list(map(lambda x:x-1, new_ls))

    return new_ls  # 返回修改后的resid_list，其实更确切应该叫seq_position_list了

# MDAnalysis部分
# 生成MDAnalysis的选择式子
def select_match_ca(mda_universe, resid_list):
    selection = ''
    for resid in resid_list:
        selection += 'resid %d or ' % resid
    selection = f'({selection[:-4]}) and name CA'
#     print(selection)
    atom_select = mda_universe.select_atoms(selection)

    return atom_select    

'''
RMSD数据分布作图所需的函数
'''
# 获得切片/分组的区间
def get_bins(df, by='rmsd', interval=10):
    '''
    df: 读取到的zdock complex decoys的信息表
    by: 划分依据，columns名称
    interval: rmsd的划分区间，默认为10
    '''
# 获得区间：从最小rmsd到最大rmsd，每隔10一个区间  e.g. [20, 30, 40, 50, 60]
    floor = (df[by].min() // interval) * interval
    floor = floor.astype('int')
    ceil = (df[by].max() // interval+ 1) * interval
    ceil = ceil.astype('int64')
    # 切片，统计区间内个数
    intervals = list(range(floor, ceil+interval, interval))

    return intervals