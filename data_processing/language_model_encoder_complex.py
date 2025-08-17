from feature import *
import os
from transformers import EsmTokenizer, EsmForMaskedLM, pipeline
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline


device1 = 'cuda:2'
# chemberta模型的读入
model_path = "/data4_large1/home_data/xyli/pretrained/ChemBERTa/ChemBERTa-77M-MTR/"
model = AutoModelWithLMHead.from_pretrained(model_path)
model.to(device1)
tokenizer = AutoTokenizer.from_pretrained(model_path)
# 去掉decoder层
model.lm_head = nn.Sequential()
chemberta_feature = pipeline('feature-extraction', model=model, tokenizer=tokenizer)  # out_dim:384

# saprot模型的读入
device2 = 'cuda:3'
model_path = "/data4_large1/home_data/xyli/pretrained/SaProt/SaProt_650M_PDB" # Note this is the directory path of SaProt, not the ".pt" file
tokenizer = EsmTokenizer.from_pretrained(model_path)
model = EsmForMaskedLM.from_pretrained(model_path)
model.to(device2)
saprot_feature = pipeline('feature-extraction', model=model, tokenizer=tokenizer)  # out_dim:440

# 两个模型无法同时放在一个cuda上，空间不够
mhc_class = 2
complex_dir = f'/data5_large/home/xyli/neoantigen/test_model/internal_test/AF3_prediction/class{mhc_class}_feature/af3_ternary'
rec_dir = f'/data5_large/home/xyli/neoantigen/test_model/internal_test/AF3_prediction/class{mhc_class}_feature/af3_rec'
lig_dir = f'/data5_large/home/xyli/neoantigen/test_model/internal_test/AF3_prediction/class{mhc_class}_feature/af3_lig'
npz_dir = f'/data5_large/home/xyli/neoantigen/test_model/internal_test/AF3_prediction/class{mhc_class}_feature/af3_lm_npz'
os.system(f'mkdir {npz_dir}')
complex_ls = os.listdir(complex_dir)
complex_ls.sort()

err_ls = []
for complex in complex_ls:
    # 拼成复合物的矩阵
    prefix = complex[:-4]
    # pmhc_pdbid, complex_name = complex.split(sep='_')
    print(f'-------processing:{prefix}--------')
    rec_pdb = os.path.join(rec_dir, complex)
    lig_pdb = os.path.join(lig_dir, complex)
#     complex_pdb = os.path.join(complex_dir, pmhc_pdbid, tcr_pdbid, 'complex', complex_name)
#     complex_pdb = os.path.join(complex_dir, complex)
    complex_filename = [filename for filename in os.listdir(complex_dir) if prefix in filename][0]
    complex_pdb = os.path.join(complex_dir, complex_filename)
    print(complex_pdb)

    try:
        complex_feature = Complex_feature(rec_fulllength_pdb=complex_pdb,
                                        rec_interface_pdb=rec_pdb,
                                        lig_fulllength_pdb=complex_pdb,
                                        lig_interface_pdb=lig_pdb,
                                        atom_feature_extractor=chemberta_feature,
                                        res_feature_extractor=saprot_feature)
        
        atom_node, atom_adj = complex_feature.get_atom_data()
        res_node, res_adj = complex_feature.get_res_data()
        save_path = os.path.join(npz_dir, prefix+'.npz')
        np.savez(save_path, 
                atom_node= atom_node, atom_adj=atom_adj, 
                res_node = res_node, res_adj=res_adj)
        
    except:
        err_ls.append(complex)
    

print(err_ls)
with open(f'/data5_large/home/xyli/neoantigen/test_model/internal_test/AF3_prediction/class{mhc_class}_feature/npz.err', 'a') as f1:
    for err in err_ls:
        f1.write(err+'\n')