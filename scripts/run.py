from run_model import run_plugin
from utils.hovernet_tools import extract_tiles_hovernet
import pandas as pd
import scanpy as sc
import torch

adata_name = 'CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma'
data_path = f'../data/{adata_name}/'
json_path = data_path + 'seg_json/pannuke_fast_mask_lvl3.json'
image_path = data_path + 'CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_tissue_image.tif'
path_ST_adata = data_path + "ST/"

#load adata
adata = sc.read_visium(path_ST_adata)

#load image dict
image_dict = torch.load(data_path + "images_dict.pt")

#load proportions
proportions_file = data_path + "C2L_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_prop.csv"
proportions = pd.read_csv(proportions_file, index_col=0)

run_plugin(adata, 
           adata_name, 
           json_path, 
           image_dict, 
           proportions, 
           batch_size=8, 
           lr=1e-5,
           agg_loss='onehot', #change
           alpha=0.5, #change
           epochs=50,
           train_size=0.5, 
           val_size=0.25, 
           out_dir='../out/model_bs8_lr1e-5_agglossonehot_alpha05_epochs50') #change

run_plugin(adata, 
           adata_name, 
           json_path, 
           image_dict, 
           proportions, 
           batch_size=8, 
           lr=1e-5,
           agg_loss='mean', #change
           alpha=0.5, #change
           epochs=50,
           train_size=0.5, 
           val_size=0.25, 
           out_dir='../out/model_bs8_lr1e-5_agglossmean_alpha05_epochs50') #change

run_plugin(adata, 
           adata_name, 
           json_path, 
           image_dict, 
           proportions, 
           batch_size=8, 
           lr=0.001,
           agg_loss='onehot', #change
           alpha=0.5, #change
           epochs=50,
           train_size=0.5, 
           val_size=0.25, 
           out_dir='../out/model_bs8_lr1e-3_agglossonehot_alpha05_epochs50') #change

run_plugin(adata, 
           adata_name, 
           json_path, 
           image_dict, 
           proportions, 
           batch_size=8, 
           lr=0.001,
           agg_loss='mean', #change
           alpha=0.5, #change
           epochs=50,
           train_size=0.5, 
           val_size=0.25, 
           out_dir='../out/model_bs8_lr1e-3_agglossmean_alpha05_epochs50') #change