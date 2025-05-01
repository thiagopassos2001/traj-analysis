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
    #     "data/json/BM_x_PA_D2_0001.json",
    #     post_processing=model.PostProcessing1)
    # result,a = model.Hd4Analysis(model.green_open_time[1])
    
    # print(result) # [result.columns[:12]]
    # print(a)

    root_path = r"C:\Users\User\Desktop\Repositórios Locais\traj-analysis"
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
    df.to_excel("data/summary/hd4_30_04_25.xlsx",index=False)

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
    df.to_excel("data/summary/hd1_30_04_25.xlsx",index=False)
