 # Libs principais
from model import *
import pandas as pd
# Controle de execução e pastas
import os
# Desativar alguns warnings
import warnings
warnings.filterwarnings('ignore')

import timeit

start_timer = timeit.default_timer()
# Notas

# # Suavização
# model = YoloMicroscopicDataProcessing()
# model.ImportFromJSON(
#     "data/json/C_x_13M_SemMotobox_D5_0001.json",
#     post_processing=model.PostProcessing1)
# model_smoothed = model.SmoothingSavGolFilter(window_length=15,polyorder=1)
# model_smoothed.to_csv("output.csv",index=False)

if __name__=="__main__":
    model = YoloMicroscopicDataProcessing()
    model.ImportFromJSON(
        "data/json/C_x_13M_SemMotobox_D5_0001.json",
        post_processing=model.PostProcessing1)
    # Suavizar  (1a vez)
    # model_smoothed = model.SmoothingSavGolFilter(window_length=15,polyorder=1)
    # model_smoothed.to_csv("tests\Smoothing_C_x_13M_SemMotobox_D5_0001.csv",index=False)

    model.df = pd.read_csv("tests\Smoothing_C_x_13M_SemMotobox_D5_0001.csv")

    id_motorcycle

    # root_path = r"C:\Users\User\Desktop\Repositórios Locais\traj-analysis"

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
    # df.to_excel("data/summary/hd1_10_05_25.xlsx",index=False)

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
    # df.to_excel("data/summary/hd4_10_05_25.xlsx",index=False)

    # df = []
    # all_files = os.listdir("data/hd_check")
    # for f in all_files:
    #     df_ = pd.read_csv(os.path.join("data/hd_check",f))
    #     df.append(df_)
    # df = pd.concat(df,ignore_index=True)
    # df.to_excel("data/summary/hd_check_10_05_25.xlsx",index=False)
stop_timer = timeit.default_timer()
count_timer = stop_timer - start_timer
print(f"Execução: {int(count_timer//60):02}:{int(count_timer%60):02} (mm:ss)")