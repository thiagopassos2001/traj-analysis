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
    mode = "run_sat_headway"

    if mode=="test3":
        root_path = "data_ignore"
        os.chdir(root_path)
        model = YoloMicroscopicDataProcessing()
        model.ImportFromJSON("data/json/79_B_2.json",post_processing=model.PostProcessing1)

        print(model.HeadwayDeltaSpeed(304,int(55*30)))
        
    if mode=="test2":
        root_path = "data_ignore"
        os.chdir(root_path)

        file_list = os.listdir("data/raw")
        valid_id = ["_".join(i.split("_")[:-2]) for i in file_list]

        df_parameter = pd.read_excel("data/Dados dos vídeos consolidados.xlsx",sheet_name='Coleta')
        df_parameter = df_parameter[df_parameter["id_voo"].isin(valid_id)]
        df_parameter = df_parameter[df_parameter["tipo_local"]=="MQ"]

        print(df_parameter["id_voo"].tolist())

        for index,row in df_parameter.iterrows():
            try:
                print("Processando",row['id_voo'])
                limite_faixa = eval(row["limite_faixa"])
                # ll = [[0,limite_faixa[-1][-1]],[1920,limite_faixa[-1][-1]]]
                # limite_faixa = [[[0,i[0]],[1920,i[0]]] for i in eval(row["limite_faixa"])]
                # limite_faixa.append(ll)
                limite_faixa = [[[0,i[0][-1]]]+i+[[1920,i[-1][-1]]] for i in limite_faixa]
                print(limite_faixa)

                RunDataProcessingFromSheetType1(
                    raw_file_path=os.path.join(f"data/raw/{row['id_voo']+"_transformed_rastreio.csv"}"),
                    file_name=row["id_voo"],
                    mpp=float(row["mpp"]),
                    flip_h=True if row["fluxo"]!="→" else False,
                    virtual_lane_lim=limite_faixa,
                    image_reference=row["img_ref"]
                )

                model = YoloMicroscopicDataProcessing()
                model.ImportFromJSON(f"data/json/{row['id_voo']}.json")
                model_smoothed = model.SmoothingSavGolFilter(window_length=15,polyorder=1)
                model_smoothed.to_csv(f"data/suavizado/{row['id_voo']}.csv",index=False)
                print("Fim",row['id_voo'])
            except Exception as e:
                print(e)
    
    if mode=="rerun":
        root_path = "data_ignore"
        os.chdir(root_path)
        model = YoloMicroscopicDataProcessing()
        model.ImportFromJSON("data/json/32_A_5.json",post_processing=model.PostProcessing1)
        print(model.df)

    if mode=="run_sat_headway":
        root_file = "data/json"
        all_files = os.listdir(root_file)

        exist_files = ["_".join(i.split("_")[2:]).split(".")[0]+".json" for i in os.listdir("data/sat_headway")]
        all_files = reversed([i for i in all_files if i not in exist_files])
        

        for f in all_files:
            print(f"Processando {f}")
            model = YoloMicroscopicDataProcessing()
            model.ImportFromJSON(os.path.join(root_file,f),post_processing=model.PostProcessing1)

            df = []
            df1 = []
            range_instant = model.green_open_time+[model.df[model.instant_column].max()-10]
            for i in range(len(range_instant)-1):
                start_instant = range_instant[i]
                last_instant = range_instant[i+1]

                try:
                    result = model.DischargeHeadwayMotorcycleAnalysis(
                        start_frame=int(model.fps*start_instant),
                        last_frame=int(model.fps*last_instant)
                    )
                    df.append(result)
                except Exception as e:
                    print("Erro no DischargeHeadwayMotorcycleAnalysis")
                    print(e)

                try:
                    result1 = model.GVCS_Type1(
                        start_frame=int(model.fps*start_instant),
                        last_frame=int(model.fps*last_instant),
                    )
                    df1.append(result1)
                except Exception as e:
                    print("Erro no GVCS_Type1")
                    print(e)
            
            try:
                df = pd.concat(df,ignore_index=True)
                df = df.drop_duplicates(subset="id_follower",keep="last")
                df.to_csv(f"data/sat_headway/sat_headway_{f.replace('json','csv')}",index=False)
            except Exception as e:
                print(f"Erro salvar aquivo sat_headway_{f}")
                print(e)

            try:
                df1 = pd.concat(df1,ignore_index=True)
                df1 = df1.drop_duplicates(subset="id",keep="last")
                df1.to_csv(f"data/geral_headway/geral_headway_{f.replace('json','csv')}",index=False)
            except Exception as e:
                print(f"Erro salvar aquivo geral_headway_{f}")
                print(e)
            
            print(f"Concluído {f}")

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