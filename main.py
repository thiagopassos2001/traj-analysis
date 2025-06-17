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
# Nota

# # Suavização
# model = YoloMicroscopicDataProcessing()
# model.ImportFromJSON(
#     "data/json/C_x_13M_SemMotobox_D5_0001.json",
#     post_processing=model.PostProcessing1)
# model_smoothed = model.SmoothingSavGolFilter(window_length=15,polyorder=1)
# model_smoothed.to_csv("output.csv",index=False)

if __name__=="__main__":
    mode = "test"

    if mode=="test2":
        root_path = "data_ignore"
        os.chdir(root_path)

        file_list = os.listdir("data/raw")
        valid_id = ["_".join(i.split("_")[:-2]) for i in file_list]

        df_parameter = pd.read_excel("Dados dos vídeos consolidados.xlsx",sheet_name='Coleta')
        df_parameter = df_parameter[df_parameter["id_voo"].isin(valid_id)]

        for index,row in df_parameter.iterrows():
            limite_faixa = eval(row["limite_faixa"])
            ll = [[0,limite_faixa[-1][-1]],[1920,limite_faixa[-1][-1]]]
            limite_faixa = [[[0,i[0]],[1920,i[0]]] for i in eval(row["limite_faixa"])]
            limite_faixa.append(ll)

            RunDataProcessingFromSheetType1(
                raw_file_path=os.path.join(f"data/raw/{row['id_voo']+"_transformed_rastreio.csv"}"),
                file_name=row["id_voo"],
                mpp=float(row["mpp"]),
                flip_h=True if row["fluxo"]!="→" else False,
                virtual_lane_lim=limite_faixa,
                image_reference=row["img_ref"]
            )
    
    if mode=="rerun":
        RunDataProcessingFromParameterType1(
            "data/json/C_x_13M_SemMotobox_D4_0004.json",
            force_processing=True)

    if mode=="test":
        model = YoloMicroscopicDataProcessing()
        model.ImportFromJSON("data/json/BM_x_PA_D2_0001.json",post_processing=model.PostProcessing1)

        df = []
        df1 = []
        range_instant = model.green_open_time+[model.df[model.instant_column].max()]
        for i in range(len(range_instant)-1):
            start_instant = range_instant[i]
            last_instant = range_instant[i+1]

            result = model.DischargeHeadwayMotorcycleAnalysis(
                start_frame=int(model.fps*start_instant),
                last_frame=int(model.fps*last_instant)
            )
            result1 = model.GVCS_Type1(
                start_frame=int(model.fps*start_instant),
                last_frame=int(model.fps*last_instant),
            )
            df.append(result)
            df1.append(result1)
        
        df = pd.concat(df,ignore_index=True)
        df = df.drop_duplicates(subset="id_follower",keep="last")
        df.to_excel("tests/Headway_Sat_BM_x_PA_D2_0001.xlsx",index=False)

        df1 = pd.concat(df1,ignore_index=True)
        df1 = df1.drop_duplicates(subset="id",keep="last")
        df1.to_excel("tests/Headway_Geral_BM_x_PA_D2_0001.xlsx",index=False)

    if mode=="processing":
        root_path = r"C:\Users\User\Desktop\Repositórios Locais\traj-analysis\data\json"
        all_files = os.listdir(root_path)
        for f in all_files:
            RunDataProcessingFromParameterType1(
                os.path.join(root_path,f),
                force_processing=True,
                )

    if mode=="run":
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
        df.to_excel("data/summary/hd1_25_05_25.xlsx",index=False)

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
        df.to_excel("data/summary/hd4_25_05_25.xlsx",index=False)

        df = []
        all_files = os.listdir("data/hd_check")
        for f in all_files:
            df_ = pd.read_csv(os.path.join("data/hd_check",f))
            df.append(df_)
        df = pd.concat(df,ignore_index=True)
        df.to_excel("data/summary/hd_check_25_05_25.xlsx",index=False)

stop_timer = timeit.default_timer()
count_timer = stop_timer - start_timer
print(f"Execução: {int(count_timer//60):02}:{int(count_timer%60):02} (mm:ss)")