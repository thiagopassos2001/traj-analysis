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
    mode = "overtaking"

    if mode=="overtaking":
        root_path = "data_ignore"
        os.chdir(root_path)

        # Início do loop
        for f in os.listdir("data/json"):
            try:
                save_file_path = f"data/collected/motorcycle_overtaking/{f.replace('.json','.csv')}"
                if not os.path.exists(save_file_path):
                    model = YoloMicroscopicDataProcessing()
                    model.ImportFromJSON(f"data/json/{f}",post_processing=model.PostProcessing1)

                    bd_support= pd.read_excel(f"data/Dados dos vídeos consolidados.xlsx",sheet_name="Vídeos")
                    bd_support = bd_support[bd_support["id_video"]==f.split(".")[0]]
                    virtual_lane_lim = [int(i) for i in bd_support.astype(str)["cod_motofaixa"].values[0].split(",")]
                    traffic_lane_list = [int(i) for i in bd_support.astype(str)["cod_faixas"].values[0].split(",")]
                    
                    model.df = model.df[model.df["traffic_lane"].isin(traffic_lane_list)]

                    motorcycle_fr2side = pd.read_csv(f"data/collected/motorcycle_fr2side/{f.replace('.json','.csv')}")

                    df_agg_motorcycle_overtaking = motorcycle_fr2side.groupby("id").agg({
                        "front":"unique",
                        "rear":"unique",
                        "left":"unique",
                        "right":"unique"
                    }).reset_index()

                    df_overtaking = {
                        "id":[],
                        "id_front":[],
                        "frame":[],
                        "OD":[],
                        "overtaking_complete":[],
                        "min_TTC":[],
                        "dv":[],
                        "H":[],
                    }

                    for index,row in df_agg_motorcycle_overtaking.iterrows():
                        for front in row["front"]:
                            if front!="nan":
                                if front in row["right"]:
                                    is_overtaking = True
                                    side = "right"
                                elif front in row["left"]:
                                    is_overtaking = True
                                    side = "left"
                                else:
                                    is_overtaking = False
                                
                                if is_overtaking:
                                    mask_follower = motorcycle_fr2side["id"]==row["id"]
                                    mask_leader_front = motorcycle_fr2side["front"]==front
                                    mask_leader_side = motorcycle_fr2side[side]==front

                                    # TTC
                                    df_overtaking_front = motorcycle_fr2side[mask_follower & mask_leader_front].copy()

                                    df_overtaking_front = df_overtaking_front.merge(model.df[["id","frame","x_instant_speed"]],on=["id","frame"],how="left")
                                    df_overtaking_front = df_overtaking_front.merge(model.df[["id","frame","x_instant_speed"]].rename(columns={"id":"front"}),on=["front","frame"],how="left",suffixes=("_follower","_leader"))

                                    df_overtaking_front["TTC"] = df_overtaking_front["dist_front"]/(df_overtaking_front["x_instant_speed_follower"]-df_overtaking_front["x_instant_speed_leader"])

                                    min_TTC = df_overtaking_front[df_overtaking_front["TTC"]>0]["TTC"].min()

                                    # H and delta_speed
                                    df_overtaking_side = motorcycle_fr2side[mask_follower & mask_leader_side].copy()

                                    df_overtaking_side = df_overtaking_side.merge(model.df[["id","frame","x_instant_speed"]],on=["id","frame"],how="left")
                                    df_overtaking_side = df_overtaking_side.merge(model.df[["id","frame","x_instant_speed"]].rename(columns={"id":side}),on=[side,"frame"],how="left",suffixes=("_follower","_leader"))

                                    df_overtaking_side["dv"] = df_overtaking_front["x_instant_speed_follower"]-df_overtaking_front["x_instant_speed_leader"]
                                    dv = df_overtaking_side["dv"].mean()
                                    H = df_overtaking_side["dist_"+side].mean()

                                    # Overtaking duration
                                    overtakink_start_frame = df_overtaking_side["frame"].min()
                                    overtakink_end_frame = df_overtaking_side["frame"].max()
                                    OD = (overtakink_end_frame-overtakink_start_frame)/30
                                    overtaking_complete = True if motorcycle_fr2side[mask_follower].copy()["frame"].max()>overtakink_end_frame else False

                                    df_overtaking["id"].append(row["id"])
                                    df_overtaking["id_front"].append(front)
                                    df_overtaking["frame"].append(overtakink_start_frame)
                                    df_overtaking["OD"].append(OD)
                                    df_overtaking["overtaking_complete"].append(overtaking_complete)
                                    df_overtaking["min_TTC"].append(min_TTC)
                                    df_overtaking["dv"].append(dv)
                                    df_overtaking["H"].append(H)

                    df_overtaking = pd.DataFrame.from_dict(df_overtaking)
                    df_overtaking.to_csv(save_file_path,index=False)

                    print("OK",f)
            except Exception as e:
                print("Buxo",f)

    if mode=="CorridorClass":
        root_path = "data_ignore"
        os.chdir(root_path)

        # Início do loop
        for f in os.listdir("data/json"):
            save_file_path = f"data/collected/motorcycle_virtual_lane/{f.replace('.json','.csv')}"
            if not os.path.exists(save_file_path):
                model = YoloMicroscopicDataProcessing()
                model.ImportFromJSON(f"data/json/{f}",post_processing=model.PostProcessing1)

                def DropDup(sample):
                    new_sample = []
                    for i in sample:
                        if i not in new_sample:
                            new_sample.append(i)
                    
                    return new_sample

                model.virtual_lane_lim = [DropDup(i) for i in model.virtual_lane_lim]

                bd_support= pd.read_excel(f"data/Dados dos vídeos consolidados.xlsx",sheet_name="Vídeos")
                bd_support = bd_support[bd_support["id_video"]==f.split(".")[0]]
                virtual_lane_lim = [int(i) for i in bd_support.astype(str)["cod_motofaixa"].values[0].split(",")]
                traffic_lane_list = [int(i) for i in bd_support.astype(str)["cod_faixas"].values[0].split(",")]
                
                model.df = model.df[model.df["traffic_lane"].isin(traffic_lane_list)]

                df_motorcycle = model.df[model.df[model.vehicle_type_column].isin(['Moto'])].sort_values([model.frame_column,model.id_column])
                df_motorcycle = df_motorcycle[["frame","id","x","y"]]
                virtual_lane_width = 1.6

                virutal_lane_group = {'Corredor Principal':virtual_lane_lim}
                virutal_lane_group["Outros Corredores"] = [i for i in  range(1,len(model.virtual_lane_lim)+1) if i not in virutal_lane_group['Corredor Principal']]

                # Qual corredor virual pertence (-1 para nenhum corredor)
                df_motorcycle['virutal_lane'] = df_motorcycle.apply(lambda row:VirtualLaneDetector(row[model.x_centroid_column],row[model.y_centroid_column],model.virtual_lane_lim,virtual_lane_width),axis=1)
                # Corredores não classificados recebem 0
                df_motorcycle['zero_temp'] = ((-df_motorcycle['virutal_lane'].isin(JoinList(list(virutal_lane_group.values())))) & (df_motorcycle['virutal_lane']!=-1))
                df_motorcycle['virutal_lane'] = df_motorcycle.apply(lambda x:x['virutal_lane'] if not x['zero_temp'] else 0,axis=1)
                df_motorcycle = df_motorcycle.drop(columns=['zero_temp'])

                # Tipo/nome do corredor
                df_motorcycle['virtual_lane_type'] = np.nan
                for key,value in virutal_lane_group.items():
                    df_motorcycle['virtual_lane_type'] = df_motorcycle.apply(lambda x:key if x['virutal_lane'] in value else x['virtual_lane_type'],axis=1)
                # Se for -1, estava na mais centralizado na faixa de tráfefo misto
                df_motorcycle['virtual_lane_type'] = df_motorcycle.apply(lambda x:'Fora do Corredor' if x['virutal_lane']==-1 else x['virtual_lane_type'],axis=1)
                # Se estava em outras faixas não avaliadas
                df_motorcycle['virtual_lane_type'] = df_motorcycle.apply(lambda x:'Outro Corredor' if x['virutal_lane']==0 else x['virtual_lane_type'],axis=1)
                df_motorcycle = df_motorcycle[['frame', 'id','virutal_lane', 'virtual_lane_type']]
                
                df_motorcycle.to_csv(save_file_path,index=False)
                model.CreateJSON(f"data/json/{f}")
                print("OK",f)
            else:
                print("OK",f,"já processado")

    if mode=="speed_profile":
        root_path = "data_ignore"
        os.chdir(root_path)

        # Início do loop
        for f in os.listdir("data/json"):
            save_file_path = f"data/collected/speed_profile/{f.replace('.json','.csv')}"
            if not os.path.exists(save_file_path):
                model = YoloMicroscopicDataProcessing()
                model.ImportFromJSON(f"data/json/{f}",post_processing=model.PostProcessing1)

                bd_support= pd.read_excel(f"data/Dados dos vídeos consolidados.xlsx",sheet_name="Vídeos")
                bd_support = bd_support[bd_support["id_video"]==f.split(".")[0]]
                virtual_lane_lim = [int(i) for i in bd_support.astype(str)["cod_motofaixa"].values[0].split(",")]
                traffic_lane_list = [int(i) for i in bd_support.astype(str)["cod_faixas"].values[0].split(",")]
                
                model.df = model.df[model.df["traffic_lane"].isin(traffic_lane_list)]

                df_agg = model.df[model.df[model.vehicle_type_column].isin(['Moto'])]
                df_agg = df_agg[df_agg['x'].between(3,model.video_width-3)].sort_values([model.frame_column,model.id_column])
                df_agg = df_agg.groupby("id")["instant_speed"].describe().reset_index(drop=False)
                df_agg["exceeded_speed_limitition"] = df_agg["max"]>(bd_support["lim_velocidade"].values[0]/3.6)
                df_agg["exceeded_speed_limitition"] = df_agg["exceeded_speed_limitition"].astype(int)

                df_agg.to_csv(save_file_path,index=False)
                print("OK",f)
            else:
                print("Já processado",f)

    if mode=="TrafficCondition":
        root_path = "data_ignore"
        os.chdir(root_path)

        # Início do loop
        for f in os.listdir("data/json"):
            save_file_path = f"data/collected/traffic_conditions/{f.replace('.json','.csv')}"
            if not os.path.exists(save_file_path):
                model = YoloMicroscopicDataProcessing()
                model.ImportFromJSON(f"data/json/{f}",post_processing=model.PostProcessing1)


                bd_support= pd.read_excel(f"data/Dados dos vídeos consolidados.xlsx",sheet_name="Vídeos")
                bd_support = bd_support[bd_support["id_video"]==f.split(".")[0]]
                virtual_lane_lim = [int(i) for i in bd_support.astype(str)["cod_motofaixa"].values[0].split(",")]
                traffic_lane_list = [int(i) for i in bd_support.astype(str)["cod_faixas"].values[0].split(",")]
                
                model.df = model.df[model.df["traffic_lane"].isin(traffic_lane_list)]

                df_agg = model.df[-model.df[model.vehicle_type_column].isin(['Moto'])].sort_values([model.frame_column,model.id_column])
                df_agg = df_agg.groupby(model.frame_column)["instant_speed"].describe().reset_index(drop=False)
                df_agg["traffic_condition"] = df_agg["mean"]<(10/3.6)
                df_agg["traffic_condition"] = -df_agg["traffic_condition"]
                df_agg["traffic_condition"] = df_agg["traffic_condition"].astype(int)

                df_agg.to_csv(save_file_path,index=False)
                print("OK",f)
            else:
                print("Já processado",f)

    if mode=="check_dir_flow":
        root_path = "data_ignore"
        os.chdir(root_path)

        if True:
            for f in os.listdir("data/json"):
                # if not os.path.exists(f"data/collected/count_flow_speed/{f.replace('.json','.csv')}"):
                try:
                    model = YoloMicroscopicDataProcessing()
                    model.ImportFromJSON(f"data/json/{f}",post_processing=model.PostProcessing1)

                    traffic_lane_list = pd.read_excel(f"data/Dados dos vídeos consolidados.xlsx",sheet_name="Coletas")
                    traffic_lane_list = traffic_lane_list[traffic_lane_list["id_coleta"].astype(str)==f.split("_")[0]]
                    traffic_lane_list = traffic_lane_list.astype(str)["cod_faixas"].values[0].split(",")
                    traffic_lane_list = [int(i) for i in traffic_lane_list]

                    df_check = model.df.copy()
                    df_check = df_check[df_check["traffic_lane"].isin(traffic_lane_list)]
                    df_check = df_check.groupby("id").agg(
                        {
                            "x":["first","last"]
                        }
                    ).reset_index(drop=False)
                    df_check.columns = ["id","first","last"]
                    df_check["diff"] = df_check["last"] - df_check["first"]
                    print(df_check["diff"].mean())
                    print("OK",f)
                except Exception as e:
                    print("Não OK",f)
                    print(f"Erro: {e}")
                # else:
                #     print(f"Arquivo {f} já existe!")

    if mode=="FR2SMapping":
        root_path = "data_ignore"
        os.chdir(root_path)

        # Início do loop
        for f in os.listdir("data/json"):
            save_file_path = f"data/collected/motorcycle_fr2side/{f.replace('.json','.csv')}"
            if not os.path.exists(save_file_path):
                model = YoloMicroscopicDataProcessing()
                model.ImportFromJSON(f"data/json/{f}",post_processing=model.PostProcessing1)

                bd_support= pd.read_excel(f"data/Dados dos vídeos consolidados.xlsx",sheet_name="Vídeos")
                bd_support = bd_support[bd_support["id_video"]==f.split(".")[0]]
                virtual_lane_lim = [int(i) for i in bd_support.astype(str)["cod_motofaixa"].values[0].split(",")]
                traffic_lane_list = [int(i) for i in bd_support.astype(str)["cod_faixas"].values[0].split(",")]
                
                model.df = model.df[model.df["traffic_lane"].isin(traffic_lane_list)]

                df_agg = model.df[model.df[model.vehicle_type_column].isin(['Moto'])]
                df_agg = df_agg.apply(lambda row:model.FR2SMapping(row["id"],row["frame"]).iloc[0],axis=1,result_type="expand")

                df_agg.to_csv(save_file_path,index=False)
                print("OK",f)
            else:
                print("Já processado",f)

    if mode=="sfd":
        root_path = "data_ignore"
        os.chdir(root_path)

        if True:
            for f in os.listdir("data/json"):
                if not os.path.exists(f"data/collected/count_flow_speed/{f.replace('.json','.csv')}"):
                    try:
                        model = YoloMicroscopicDataProcessing()
                        model.ImportFromJSON(f"data/json/{f}",post_processing=model.PostProcessing1)

                        traffic_lane_list = pd.read_excel(f"data/Dados dos vídeos consolidados.xlsx",sheet_name="Coletas")
                        traffic_lane_list = traffic_lane_list[traffic_lane_list["id_coleta"].astype(str)==f.split("_")[0]]
                        traffic_lane_list = traffic_lane_list.astype(str)["cod_faixas"].values[0].split(",")
                        traffic_lane_list = [int(i) for i in traffic_lane_list]

                        result = model.SpeedFlowDensityAgg(step=3600,traffic_lane_list=traffic_lane_list)
                        result = result.reset_index(level=[0],drop=True).T
                        result.to_csv(f"data/collected/count_flow_speed/{f.replace('.json','.csv')}")

                        print("OK",f)
                    except Exception as e:
                        print("Não OK",f)
                        print(f"Erro: {e}")
                else:
                    print(f"Arquivo {f} já existe!")

    if mode=="count_by_type_lane":
        root_path = "data_ignore"
        os.chdir(root_path)

        file_list = os.listdir("data/raw")

        for i in file_list:
            try:
                # model = YoloMicroscopicDataProcessing()
                # model.ImportFromJSON(f"data/json/{i}",post_processing=model.PostProcessing1)
                df = pd.read_csv(f"data/raw/{i}")

                df_agg = df.groupby("id").agg({"faixa":pd.Series.mode,"tipo":pd.Series.mode})
                df_agg = df_agg.reset_index(drop=True)
                df_agg["faixa"] = df_agg["faixa"].astype("str")
                df_agg["tipo"] = df_agg["tipo"].astype("str")
                df_agg["contador"] = 1
                df_agg = df_agg.groupby(["faixa","tipo"]).agg({"contador":"sum"})
                df_agg = df_agg.reset_index(drop=False)
                df_agg.insert(0,"file",i)
                df_agg["instante"] = df["instante"].max()/3600

                df_agg.to_csv(f"data/count_by_type_lane/{i}",index=False)
                print(f"{i} OK")
            except Exception as e:
                print(f"{i} deu errro",e)

    if mode=="FatEq":
        root_file = "data/json"
        all_files = os.listdir(root_file)

        exist_files = ["_".join(i.split("_")[2:]).split(".")[0]+".json" for i in os.listdir("data/equivalence_factor") if i!="V1"]
        all_files = [i for i in all_files if i not in exist_files]
        
        for f in all_files:
            print(f"Processando {f}")
            model = YoloMicroscopicDataProcessing()
            model.ImportFromJSON(os.path.join(root_file,f),post_processing=model.PostProcessing1)

            df = []
            try:
                range_instant = model.green_open_time+[model.df[model.instant_column].max()-10]
                for i in range(len(range_instant)-1):
                    start_instant = range_instant[i]
                    last_instant = range_instant[i+1]

                    try:
                        result = model.EquivalenceFactor(
                            int(round(start_instant*model.fps,0)),
                            int(round(last_instant*model.fps,0))
                        )
                        df.append(result)
                    except Exception as e:
                        print("Erro no Fator de Equivalencia")
                        print(e)
                try:
                    df = pd.concat(df,ignore_index=True)
                    df.to_csv(f"data/equivalence_factor/fator_equivalencia_{f.replace('json','csv')}",index=False)
                except Exception as e:
                    print(f"Erro salvar aquivo fator_equivalencia_{f}")
                    print(e)
                
                print(f"Concluído {f}")
            except:
                print("Erro Geral")

    if mode=="concat":
        # Concatenar resumo
        output_folder = 'data_ignore/data/collected/count_flow_speed'
        df = []
        all_files = os.listdir(output_folder)
        for f in all_files:
            df_ = pd.read_csv(os.path.join(output_folder,f),index_col="Unnamed: 0")

            df_agg = pd.DataFrame()
            df_agg["file"] = [f]
            print(f)

            sum_count = df_.loc["count"].sum()
            mean_speed = sum(df_.loc["count"]*df_.loc["speed"])/sum_count
            
            df_agg["count"] = sum_count
            df_agg["speed"] = mean_speed
            df_agg["motorcycle_count"] = df_.loc["count","Moto"]
            df_agg["motorcycle_speed"] = df_.loc["speed","Moto"]
            df_agg["motorcycle_perc"] = df_agg["motorcycle_count"]/df_agg["count"]

            df.append(df_agg)
            print(f"OK {f}")
        df = pd.concat(df,ignore_index=True)
        df.to_excel("data_ignore/data/collected/count_flow_speed.xlsx",index=False)

    if mode=="test3":
        root_path = "data_ignore"
        os.chdir(root_path)
        df = pd.read_csv("data/collected/count_flow_speed/79_B_2.csv",index_col="Unnamed: 0")
        df_agg = pd.DataFrame()
        df_agg["file"] = ["a"]

        sum_count = df.loc["count"].sum()
        mean_speed = sum(df.loc["count"]*df.loc["speed"])/sum_count
        
        df_agg["count"] = sum_count
        df_agg["speed"] = mean_speed
        df_agg["motorcycle_count"] = df.loc["count","Moto"]
        df_agg["motorcycle_speed"] = df.loc["speed","Moto"]
        
        print(df_agg)

    if mode=="test2":
        root_path = "data_ignore"
        os.chdir(root_path)

        file_list = os.listdir("data/raw")
        valid_id = ["_".join(i.split("_")[:-2]) for i in file_list]

        df_parameter = pd.read_excel("data/Dados dos vídeos consolidados.xlsx",sheet_name='Vídeos')
        df_parameter = df_parameter[df_parameter["id_video"].isin(valid_id)]
        df_parameter = df_parameter[df_parameter["id_coleta"].isin([62])] # 62,

        print(df_parameter["id_video"].tolist())

        for index,row in df_parameter.iterrows():
            if not os.path.exists(f"data/processed/{row['id_video']+'.csv'}"):
                try:
                    print("Processando",row['id_video'])
                    lim_lane_mode = "2"
                    limite_faixa = eval(row["limite_faixa"])

                    if not lim_lane_mode in ["1","2"]:
                        raise ValueError(f"{lim_lane_mode} inválido")

                    if lim_lane_mode=="1":
                        ll = [[0,limite_faixa[-1][-1]],[1920,limite_faixa[-1][-1]]]
                        limite_faixa = [[[0,i[0]],[1920,i[0]]] for i in eval(row["limite_faixa"])]
                        limite_faixa.append(ll)
                    if lim_lane_mode=="2":
                        limite_faixa = [[[0,i[0][-1]]]+i+[[1920,i[-1][-1]]] for i in limite_faixa]
                    
                    print(limite_faixa)
                    print("FLIPH",row["fluxo"],True if row["fluxo"]!="→" else False)

                    RunDataProcessingFromSheetType1(
                        raw_file_path=os.path.join(f"data/raw/{row['id_video']+"_transformed_rastreio.csv"}"),
                        file_name=row["id_video"],
                        mpp=float(row["mpp"]),
                        flip_h=True if row["fluxo"]!="→" else False,
                        virtual_lane_lim=limite_faixa,
                        image_reference=row["img_ref"]
                    )

                    model = YoloMicroscopicDataProcessing()
                    model.ImportFromJSON(f"data/json/{row['id_video']}.json")
                    model_smoothed = model.SmoothingSavGolFilter(window_length=15,polyorder=1)
                    model_smoothed.to_csv(f"data/suavizado/{row['id_video']}.csv",index=False)
                    print("Fim",row['id_video'])
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