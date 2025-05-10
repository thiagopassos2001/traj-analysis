 # Libs principais
from model import *
import pandas as pd
# Controle de execução e pastas
import os
# Desativar alguns warnings
import warnings
warnings.filterwarnings('ignore')

if __name__=="__main__":
    # model = YoloMicroscopicDataProcessing()
    # model.ImportFromJSON(
    #     "data/json/BM_x_PA_D2_0009.json",
    #     post_processing=model.PostProcessing1)
    # # result = model.Hd1Analysis(model.green_open_time[0])#[0]
    # print(model.image_reference,model.motobox_end_section,model.mpp,model.df["p2xbb"].max(),model.df["p2xbb"].max()/model.mpp)
    # print(model.df[model.df["id"]==110])
    # result = model.HdFromEndMWA(132,int(30*73.1))
    
    # print(result) # [result.columns[:12]]

    root_path = r"C:\Users\User\Desktop\Repositórios Locais\traj-analysis"

    output_folder = "data/hd1"
    run = Run()
    run.WorkflowPattern(
        root_path=root_path,
        output_folder=output_folder,
        prefix="Hd1_",
        func=RunHd1Analysis)
    
    # Concatenar resumo
    df = []
    all_files = os.listdir(output_folder)
    for f in all_files:
        df_ = pd.read_csv(os.path.join(output_folder,f))
        df.append(df_)
    df = pd.concat(df,ignore_index=True)
    df.to_excel("data/summary/hd1_10_05_25.xlsx",index=False)

    output_folder = "data/hd4"
    run = Run()
    run.WorkflowPattern(
        root_path=root_path,
        output_folder=output_folder,
        prefix="Hd4_",
        func=RunHd4Analysis)
    
    # Concatenar resumo
    df = []
    all_files = os.listdir(output_folder)
    for f in all_files:
        df_ = pd.read_csv(os.path.join(output_folder,f))
        df.append(df_)
    df = pd.concat(df,ignore_index=True)
    df.to_excel("data/summary/hd4_10_05_25.xlsx",index=False)

    df = []
    all_files = os.listdir("data/hd_check")
    for f in all_files:
        df_ = pd.read_csv(os.path.join("data/hd_check",f))
        df.append(df_)
    df = pd.concat(df,ignore_index=True)
    df.to_excel("data/summary/hd_check_10_05_25.xlsx",index=False)
