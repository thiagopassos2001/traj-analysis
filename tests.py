from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import json
import os

# Corrigir raw file path
root_json_path = "data/json"

all_files = os.listdir(root_json_path)
count = 1
for file_name  in all_files:
    file_path = os.path.join(root_json_path,file_name)
    raw_file_path = "data/raw"+"/"+file_name.replace(".json","_transformed_rastreio.csv")

    with open(file_path,encoding="utf-8",errors="ignore") as f:
        cfg = json.load(f)

    df = pd.read_csv(raw_file_path)

    old_pattern = True if 'x_esquerda' in df.columns.tolist() else False

    if not old_pattern:
        df = df.rename(columns={'x1':"x",})
    else:
        df = df.rename(columns={'x_esquerda':"x"})
    
    df = df[(df["tipo"]=="Carro") & (df["faixa"]!=0)]
    df = df.sort_values("instante")
    df_agg = df.groupby("id").agg({"x":["first","last"]}).reset_index(drop=False)
    df_agg.columns = df_agg.columns.droplevel(0)
    mean_x_dir = int((df_agg["last"]-df_agg["first"]).mean())

    cfg["flip_v"] = False
    cfg["flip_h"] = True if mean_x_dir<0 else False
    cfg["raw_file"] = raw_file_path
    
    # Atualiza o json
    with open(file_path,'w',encoding="utf-8",errors="ignore") as f:  
        json.dump(cfg,f,indent=4)

    print(count,file_path,mean_x_dir,f"FlipH = {True if mean_x_dir<0 else False}")
    count = count + 1

