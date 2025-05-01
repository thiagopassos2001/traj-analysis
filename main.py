 # Libs principais
from model import *
import pandas as pd
# Controle de execução e pastas
import os
# Desativar alguns warnings
import warnings
warnings.filterwarnings('ignore')

if __name__=="__main__":
    
    model = YoloMicroscopicDataProcessing()
    model.ImportFromJSON(
        "data/json/C_x_13M_SemMotobox_D5_0003.json",
        post_processing=model.PostProcessing1)
    result = model.GroupVechiclesCrossingSection()
    
    print(result)

    # root_path = r"C:\Users\User\Desktop\Repositórios Locais\traj-analysis"
    # output_folder = "data/hd4"
    # run = Run()
    # run.WorkflowPattern(
    #     root_path=root_path,
    #     output_folder=output_folder,
    #     prefix="Hd4_",
    #     func=RunHd4Analysis)
    
    # # Concatenar resumo
    # df = []
    # all_files = os.listdir(output_folder)
    # for f in all_files:
    #     df_ = pd.read_csv(os.path.join(output_folder,f))
    #     df.append(df_)
    # df = pd.concat(df,ignore_index=True)
    # df.to_excel("data/summary/hd4_30_04_25.xlsx",index=False)

    # output_folder = "data/hd1"
    # run = Run()
    # run.WorkflowPattern(
    #     root_path=root_path,
    #     output_folder=output_folder,
    #     prefix="Hd1_",
    #     func=RunHd1Analysis)
    
    # # Concatenar resumo
    # df = []
    # all_files = os.listdir(output_folder)
    # for f in all_files:
    #     df_ = pd.read_csv(os.path.join(output_folder,f))
    #     df.append(df_)
    # df = pd.concat(df,ignore_index=True)
    # df.to_excel("data/summary/hd1_30_04_25.xlsx",index=False)
