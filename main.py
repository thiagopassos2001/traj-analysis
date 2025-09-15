 # Libs principais
from model import *
from TTC_model import *
import pandas as pd
# Controle de execução e pastas
import os
# Desativar alguns warnings
import warnings
warnings.filterwarnings('ignore')
from scipy.signal import savgol_filter
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
    mode = "speed_x_deltaV"

    if mode=="speed_x_deltaV":
        root_path = "data_ignore"
        os.chdir(root_path)

        # Início do loop
        for f in os.listdir("data/json"):
            try:
                save_file_path = f"data/collected/speed_x_deltaV/{f.replace('.json','.csv')}"
                if not os.path.exists(save_file_path):
                    model = YoloMicroscopicDataProcessing()
                    model.ImportFromJSON(f"data/json/{f}",post_processing=model.PostProcessing1)

                    bd_support= pd.read_excel(f"data/Dados dos vídeos consolidados.xlsx",sheet_name="Vídeos")
                    bd_support = bd_support[bd_support["id_video"]==f.split(".")[0]]
                    bd_support = bd_support.merge(pd.read_excel(f"data/Dados dos vídeos consolidados.xlsx",sheet_name="Coletas")[["id_coleta","cod_motofaixa","cod_faixas","faixa_1","faixa_2"]],on="id_coleta",how="left")
                    virtual_lane_lim = [int(i) for i in bd_support.astype(str)["cod_motofaixa"].values[0].split(",")]
                    traffic_lane_list = [int(i) for i in bd_support.astype(str)["cod_faixas"].values[0].split(",")]
                    
                    model.df = model.df[model.df["traffic_lane"].isin(traffic_lane_list)]

                    motorcycle_fr2side = pd.read_csv(f"data/collected/motorcycle_fr2side/{f.replace('.json','.csv')}")
                    motorcycle_fr2side = motorcycle_fr2side.merge(model.df[["id","frame","x_instant_speed"]],on=["id","frame"],how="left").rename(columns={"x_instant_speed":"x_instant_speed_motorcycle"})
                    motorcycle_fr2side = motorcycle_fr2side.rename(columns={"id":"id_motorcycle","front":"id"}).merge(model.df[["id","frame","x_instant_speed"]],on=["id","frame"],how="left").rename(columns={"x_instant_speed":"x_instant_speed_front","id":"front","id_motorcycle":"id",})
                    motorcycle_fr2side["deltaV"] = motorcycle_fr2side["x_instant_speed_motorcycle"] - motorcycle_fr2side["x_instant_speed_front"]
                    motorcycle_fr2side["TTC"] = motorcycle_fr2side["dist_front"]/motorcycle_fr2side["deltaV"]

                    motorcycle_fr2side.to_csv(save_file_path,index=False)

                    print("OK",f)
            except Exception as e:
                print("Buxo",f)


    if mode=="width_lane":
        root_path = "data_ignore"
        os.chdir(root_path)
        df = []
        if True:
            for f in os.listdir("data/json"):
                if not os.path.exists(f"data/collected/width_lane_12/{f.replace('.json','.csv')}"):
                    try:
                        model = YoloMicroscopicDataProcessing()
                        model.ImportFromJSON(f"data/json/{f}",post_processing=model.PostProcessing1)

                        bd_support= pd.read_excel(f"data/Dados dos vídeos consolidados.xlsx",sheet_name="Vídeos")
                        bd_support = bd_support[bd_support["id_video"]==f.split(".")[0]]
                        bd_support = bd_support.merge(pd.read_excel(f"data/Dados dos vídeos consolidados.xlsx",sheet_name="Coletas")[["id_coleta","cod_motofaixa","cod_faixas","faixa_1","faixa_2"]],on="id_coleta",how="left")
                        virtual_lane_lim = [int(i) for i in bd_support.astype(str)["cod_motofaixa"].values[0].split(",")]
                        traffic_lane_list = [int(i) for i in bd_support.astype(str)["cod_faixas"].values[0].split(",")]
                        
                        model.df = model.df[model.df["traffic_lane"].isin(traffic_lane_list)]
                        x = model.video_width/2

                        first_lane = bd_support["faixa_1"].astype(int).values[0]
                        fl1 = PolygonalToFunction(model.virtual_lane_lim[first_lane-1])
                        fl2 = PolygonalToFunction(model.virtual_lane_lim[first_lane])
                        wl1 = fl2(x) - fl1(x)

                        last_lane = bd_support["faixa_2"].astype(int).values[0]
                        ll1 = PolygonalToFunction(model.virtual_lane_lim[last_lane-1])
                        ll2 = PolygonalToFunction(model.virtual_lane_lim[last_lane])
                        wl2 = ll2(x) - ll1(x)

                        df_width_lane = pd.DataFrame()
                        df_width_lane["id_video"] = [f.replace('.json','')]
                        df_width_lane["largura_faixa_1"] = [wl1]
                        df_width_lane["largura_faixa_2"] = [wl2]

                        df.append(df_width_lane)
                        
                        print("OK",f)
                    except Exception as e:
                        print("Não OK",f)
                        print(f"Erro: {e}")
                else:
                    print(f"Arquivo {f} já existe!")
        pd.concat(df,ignore_index=True).to_excel(f"data/collected/width_lane_12.xlsx",index=False)

    if mode=="agg_sfd":
        root_path = "data_ignore"
        os.chdir(root_path)

        def AggValues(ff,lf,df,df1):
            delta_t = ((lf-ff)/30)/3600
            df = df[df["frame"].between(ff,lf)]
            df_agg = {}
            
            new_cols_flow = [i.replace("_count_","_flow_") for i in df.columns if "count" in i]
            new_cols_speed = [i.replace("_count_","_speed_") for i in df.columns if "count" in i]
            new_cols_density = [i.replace("_count_","_density_") for i in df.columns if "count" in i]
            
            for col in df.columns: 
                if "count" in col:
                    df_agg[col.replace("_count_","_flow_")] = [len(df1[(df1["frame"].between(ff,lf)) & (df1["vehicle_type"]==col.split("_")[-1]) & (df1["traffic_lane"]==int(col.split("_")[-2]))])/delta_t]
                    df_agg[col.replace("_count_","_speed_")] = [df[col.replace("_count_","_mean_")].mean()*3.6]
                    df_agg[col.replace("_count_","_density_")] = [(df[col]/df["lenght"]).mean()*1000]
            
            df_agg = pd.DataFrame.from_dict(df_agg)[new_cols_flow+new_cols_speed+new_cols_density].iloc[0].tolist()

            return df_agg

        if True:
            for f in os.listdir("data/json"):
                if not os.path.exists(f"data/collected/agg_sfd/{f.replace('.json','.csv')}"):
                    try:
                        model = YoloMicroscopicDataProcessing()
                        model.ImportFromJSON(f"data/json/{f}",post_processing=model.PostProcessing1)

                        bd_support= pd.read_excel(f"data/Dados dos vídeos consolidados.xlsx",sheet_name="Vídeos")
                        bd_support = bd_support[bd_support["id_video"]==f.split(".")[0]]
                        virtual_lane_lim = [int(i) for i in bd_support.astype(str)["cod_motofaixa"].values[0].split(",")]
                        traffic_lane_list = [int(i) for i in bd_support.astype(str)["cod_faixas"].values[0].split(",")]
                        
                        model.df = model.df[model.df["traffic_lane"].isin(traffic_lane_list)]

                        section = shapely.LineString([[model.video_width*0.5,model.video_heigth],[model.video_width*0.5,0]])

                        # Veículos cruzando a seção
                        df1 = model.GroupVechiclesCrossingSection(section=section)

                        df = pd.read_csv(f"data/collected/count_speed_density_by_lane_vehicle/{f.replace('.json','.csv')}").fillna(0)
                        df_frame = pd.read_csv(f"data/collected/frame_reference/{f.replace('.json','.csv')}")
                        
                        new_cols_flow = [i.replace("x_instant_speed_count_","flow_") for i in df.columns if "count" in i]
                        new_cols_speed = [i.replace("x_instant_speed_count_","speed_") for i in df.columns if "count" in i]
                        new_cols_density = [i.replace("x_instant_speed_count_","density_") for i in df.columns if "count" in i]

                        df_frame[new_cols_flow+new_cols_speed+new_cols_density] = df_frame.apply(lambda row:AggValues(row["min"],row["max"],df,df1),axis=1,result_type="expand")
                        df_frame.to_csv(f"data/collected/agg_sfd/{f.replace('.json','.csv')}",index=False)
                        print("OK",f)
                    except Exception as e:
                        print("Não OK",f)
                        print(f"Erro: {e}")
                else:
                    print(f"Arquivo {f} já existe!")

    if mode=="speed_profile2":
        root_path = "data_ignore"
        os.chdir(root_path)

        if True:
            for f in os.listdir("data/json"):
                if not os.path.exists(f"data/collected/frame_reference/{f.replace('.json','.csv')}"):
                    try:
                        model = YoloMicroscopicDataProcessing()
                        model.ImportFromJSON(f"data/json/{f}",post_processing=model.PostProcessing1)

                        # traffic_lane_list = pd.read_excel(f"data/Dados dos vídeos consolidados.xlsx",sheet_name="Coletas")
                        # traffic_lane_list = traffic_lane_list[traffic_lane_list["id_coleta"].astype(str)==f.split("_")[0]]
                        # traffic_lane_list = traffic_lane_list.astype(str)["cod_faixas"].values[0].split(",")
                        # traffic_lane_list = [int(i) for i in traffic_lane_list]

                        # model.df = model.df[model.df[model.traffic_lane_column].isin(traffic_lane_list)]

                        result = model.df.groupby("id")["frame"].describe().reset_index(drop=False)

                        result.to_csv(f"data/collected/frame_reference/{f.replace('.json','.csv')}",index=False)
                        print("OK",f)
                    except Exception as e:
                        print("Não OK",f)
                        print(f"Erro: {e}")
                else:
                    print(f"Arquivo {f} já existe!")

    if mode=="sfd_por_veiculo_faixa":
        root_path = "data_ignore"
        os.chdir(root_path)

        if True:
            for f in os.listdir("data/json"):
                if not os.path.exists(f"data/collected/count_speed_density_by_lane_vehicle/{f.replace('.json','.csv')}"):
                    try:
                        model = YoloMicroscopicDataProcessing()
                        model.ImportFromJSON(f"data/json/{f}",post_processing=model.PostProcessing1)

                        # traffic_lane_list = pd.read_excel(f"data/Dados dos vídeos consolidados.xlsx",sheet_name="Coletas")
                        # traffic_lane_list = traffic_lane_list[traffic_lane_list["id_coleta"].astype(str)==f.split("_")[0]]
                        # traffic_lane_list = traffic_lane_list.astype(str)["cod_faixas"].values[0].split(",")
                        # traffic_lane_list = [int(i) for i in traffic_lane_list]

                        # model.df = model.df[model.df[model.traffic_lane_column].isin(traffic_lane_list)]

                        result = model.SpeedCountByFrame()
                        result.to_csv(f"data/collected/count_speed_density_by_lane_vehicle/{f.replace('.json','.csv')}")
                        print("OK",f)
                    except Exception as e:
                        print("Não OK",f)
                        print(f"Erro: {e}")
                else:
                    print(f"Arquivo {f} já existe!")

    if mode=="TTC":
        root_path = "data_ignore"
        os.chdir(root_path)

        # Início do loop
        for f in os.listdir("data/json"):
            try:
                save_file_path = f"data/collected/TTC_dir_n_suav_v2/{f.replace('.json','.csv')}"
                if not os.path.exists(save_file_path):
                    model = YoloMicroscopicDataProcessing()
                    model.ImportFromJSON(f"data/json/{f}",post_processing=model.PostProcessing1)

                    bd_support= pd.read_excel(f"data/Dados dos vídeos consolidados.xlsx",sheet_name="Vídeos")
                    bd_support = bd_support[bd_support["id_video"]==f.split(".")[0]]
                    virtual_lane_lim = [int(i) for i in bd_support.astype(str)["cod_motofaixa"].values[0].split(",")]
                    traffic_lane_list = [int(i) for i in bd_support.astype(str)["cod_faixas"].values[0].split(",")]
                    
                    model.df = model.df[model.df["traffic_lane"].isin(traffic_lane_list)]

                    crop_limits_by_vehicle_class = {
                        'Carro':(5.8,model.video_width-5.8),
                        'Moto':(2.5,model.video_width-2.5),
                        'Onibus':(15,model.video_width-15),
                        'Caminhao':(9.1,model.video_width-9.1),
                        'Van':(5.8,model.video_width-5.8),
                        'Bicicleta':(2.5,model.video_width-2.5),
                    }

                    # Alterar a frequência da amostra de 30 -> 10 fps
                    min_frame = model.df[model.frame_column].min()
                    max_frame = model.df[model.frame_column].max()
                    list_frames = list(range(min_frame,max_frame+1,3))
                    model.df = model.df[model.df[model.frame_column].isin(list_frames)]

                    # Estimativa da direção do veículo
                    # Calculado por meio da direção média do frame i-1 e i+1
                    new_model = pd.DataFrame()
                    parameter_direction_x = model.x_centroid_column
                    parameter_direction_y = model.y_centroid_column

                    for vehicle_id in model.df[model.id_column].unique():
                        vehicle = model.df[model.df[model.id_column]==vehicle_id].sort_values(model.frame_column)
                        vehicle_type = vehicle[model.vehicle_type_column].value_counts().idxmax()

                        # Remover dados das extreminades do vídeo, dependendo da classe
                        vehicle = vehicle[(vehicle[model.x_head_column]>=crop_limits_by_vehicle_class[vehicle_type][0]) & (vehicle[model.x_tail_column]<=crop_limits_by_vehicle_class[vehicle_type][-1])]
                        n_sample = len(vehicle)

                        if n_sample>2:
                            # Direção do frame i é a direção média do frame i-1 e i + 1
                            direction_x = [vehicle[parameter_direction_x].iloc[i+1]-vehicle[parameter_direction_x].iloc[i-1] for i in range(1,n_sample-1)]
                            direction_x = direction_x[:1] + direction_x + direction_x[-1:]

                            direction_y = [vehicle[parameter_direction_y].iloc[i+1]-vehicle[parameter_direction_y].iloc[i-1] for i in range(1,n_sample-1)]
                            direction_y = direction_y[:1] + direction_y + direction_y[-1:]

                            # Ajuste de direção indeterminada
                            # Se a soma das direções x e y em metro for muito baixa
                            # Ignora essa métrica e pega a anterior, pois pode tratar-se de uma oscilação aleatória, quando estiver parado

                            threshold_direction_sum = 0.60
                            for i in range(n_sample):
                                if i>3:
                                    dir_x = abs(direction_x[i])
                                    dir_y = abs(direction_y[i])
                                    direction_x[i] = direction_x[i] if dir_x+dir_y>threshold_direction_sum else np.mean(direction_x[i-3:i])
                                    direction_y[i] = direction_y[i] if dir_x+dir_y>threshold_direction_sum else np.mean(direction_y[i-3:i])

                            if False:
                                vehicle['direction_x'] = savgol_filter(direction_x,window_length=5,polyorder=1)
                                vehicle['direction_y'] = savgol_filter(direction_y,window_length=5,polyorder=1)
                            else:
                                vehicle['direction_x'] = direction_x
                                vehicle['direction_y'] = direction_y

                            new_model = pd.concat([new_model,vehicle],ignore_index=True)
                        else:
                            pass

                    model.df = new_model.sort_values(by=[model.frame_column,model.id_column])

                    # Normalizar vetor
                    vector = model.df.apply(lambda x:np.array([x['direction_x'],x['direction_y'],0]),axis=1)
                    magnitude = vector.apply(lambda x:np.linalg.norm(x))
                    normalized_vector = vector / magnitude
                    # Suavizar vetor normalizado
                    model.df['direction_norm_x'] = normalized_vector.apply(lambda x:x[0])
                    model.df['direction_norm_y'] = normalized_vector.apply(lambda x:x[1])

                    # Angulo
                    model.df['degree'] = model.df.apply(lambda x:np.degrees(np.arctan2(x['direction_norm_y'],(x['direction_norm_x'] if x['direction_norm_x']!=0 else 0.0000000000001))),axis=1)

                    # Dimensões nominais dos veículos
                    # Mediana das dimensões de cada veículo em particular
                    df_agg_vehicle_size = model.df[model.df['degree'].abs()<=5].groupby(model.id_column)[[model.vehicle_length_column,model.vehicle_width_column]].median().reset_index(drop=False)
                    df_agg_vehicle_size = df_agg_vehicle_size.rename(columns={
                        model.vehicle_length_column:model.vehicle_length_column+'_median',
                        model.vehicle_width_column: model.vehicle_width_column+'_median',
                    })
                    # Associar dimensões nominais aos veículos por id
                    model.df = model.df.merge(df_agg_vehicle_size,on=model.id_column,how='left')
                    # Casos em que o veículo foi totalmente inclinado, associa a mediana na amostra por tipo de veículo
                    vehicle_size_median = model.df[model.df['degree'].abs()<=5].groupby(model.vehicle_type_column)[[model.vehicle_length_column,model.vehicle_width_column]].median().T.to_dict()
                    model.df[model.vehicle_length_column+'_median'] = model.df[model.vehicle_length_column+'_median'].fillna(0)
                    model.df[model.vehicle_width_column+'_median'] = model.df[model.vehicle_width_column+'_median'].fillna(0)

                    model.df[model.vehicle_length_column+'_median'] = model.df.apply(lambda x:x[model.vehicle_length_column+'_median'] if x[model.vehicle_length_column+'_median']!=0 else vehicle_size_median[x[model.vehicle_type_column]][model.vehicle_length_column],axis=1)
                    model.df[model.vehicle_width_column+'_median'] = model.df.apply(lambda x:x[model.vehicle_width_column+'_median'] if x[model.vehicle_width_column+'_median']!=0 else vehicle_size_median[x[model.vehicle_type_column]][model.vehicle_width_column],axis=1)

                    model.df[model.vehicle_length_column+'_median'] = model.df[model.vehicle_length_column+'_median'] - 0.1
                    model.df[model.vehicle_width_column+'_median'] = model.df[model.vehicle_width_column+'_median'] - 0.1

                    # Area superficial
                    model.df['superficial_area_median'] = model.df[model.vehicle_length_column+'_median']*model.df[model.vehicle_width_column+'_median']

                    # Cálulo do TTC
                    # Dados no formato da função do TTC
                    # df_TTC = pd.DataFrame(columns=['x_i',
                    #  'y_i',
                    #  'vx_i',
                    #  'vy_i',
                    #  'hx_i',
                    #  'hy_i',
                    #  'length_i',
                    #  'width_i',
                    #  'x_j',
                    #  'y_j',
                    #  'vx_j',
                    #  'vy_j',
                    #  'hx_j',
                    #  'hy_j',
                    #  'length_j',
                    #  'width_j'])

                    # Velocidades são relativas ao sistema de coordenadas cartesiano
                    # No TTC, é o sistema padrão (+ p/ a direita e p/ cima)
                    # Logo, a coordenada do centroide y, velocidade y e vetor y tem que ser ajustadas (Modelo de visão computacional)
                    # Comprimento é o tamanho do box no eixo x
                    # Largura é o comprimento do box no eixo y
                    # Adotar parão de hx = 1 e hy = 0 (teste)

                    # Armazena todos os TTCs calculados ao longo dos frames
                    # Armazena todos os TTCs calculados ao longo dos frames
                    df_TTC = []
                    for t in list_frames:
                        # Veículos no instante t
                        df_analysis = model.df[model.df[model.frame_column]==t]
                        # Ajuste das coordenadas y (modelo CV)
                        df_analysis[model.y_centroid_column] = model.video_heigth - df_analysis[model.y_centroid_column]
                        df_analysis[model.y_instant_speed_column] = -df_analysis[model.y_instant_speed_column]
                        df_analysis['direction_norm_y'] = -df_analysis['direction_norm_y']

                        if len(df_analysis)>1:
                            # Pares de conflitos no instante t
                            df_TTC_t = pd.DataFrame(columns=['id_i'])

                            for id_vehicle_i in df_analysis[model.id_column]:
                                # Veículo "i" do conflito
                                vehicle_i = df_analysis[df_analysis[model.id_column]==id_vehicle_i]
                                # Demais veículos "j" que podem estar em conflito com o veículo de referencia
                                vehicle_other = df_analysis[df_analysis[model.id_column]!=id_vehicle_i]

                                # Evitar o cáluclo de pares duplicados
                                # Se o veículo "j" nesse instante já tiver sido computado como "i" em
                                # Alguma iteração anterior, esse conflito potencial já foi computado
                                # Portanto esse veículo não é mais habilitado para ser "j" pois já foi "i"
                                vehicle_other = vehicle_other[-vehicle_other[model.id_column].isin(df_TTC_t['id_i'])]

                                # Dados do veículo "i"
                                df_TTC_i = pd.DataFrame()
                                df_TTC_i['id_i'] = [id_vehicle_i]*len(vehicle_other)
                                df_TTC_i['vehicle_type_i'] = [vehicle_i[model.vehicle_type_column].iloc[0]]*len(vehicle_other)
                                df_TTC_i['x_i'] = [vehicle_i[model.x_centroid_column].iloc[0]]*len(vehicle_other)
                                df_TTC_i['y_i'] = [vehicle_i[model.y_centroid_column].iloc[0]]*len(vehicle_other)
                                df_TTC_i['vx_i'] = [vehicle_i[model.x_instant_speed_column].iloc[0]]*len(vehicle_other)
                                df_TTC_i['vy_i'] = [vehicle_i[model.y_instant_speed_column].iloc[0]]*len(vehicle_other)
                                df_TTC_i['hx_i'] = [vehicle_i['direction_norm_x'].iloc[0]]*len(vehicle_other)
                                df_TTC_i['hy_i'] = [vehicle_i['direction_norm_y'].iloc[0]]*len(vehicle_other)
                                df_TTC_i['length_i'] = [vehicle_i[model.vehicle_length_column+'_median'].iloc[0]]*len(vehicle_other)
                                df_TTC_i['width_i'] = [vehicle_i[model.vehicle_width_column+'_median'].iloc[0]]*len(vehicle_other)

                                # Dados dos veículos "j"
                                df_TTC_i['id_j'] = vehicle_other[model.id_column].values
                                df_TTC_i['vehicle_type_j'] = vehicle_other[model.vehicle_type_column].values
                                df_TTC_i['x_j'] = vehicle_other[model.x_centroid_column].values
                                df_TTC_i['y_j'] = vehicle_other[model.y_centroid_column].values
                                df_TTC_i['vx_j'] = vehicle_other[model.x_instant_speed_column].values
                                df_TTC_i['vy_j'] = vehicle_other[model.y_instant_speed_column].values
                                df_TTC_i['hx_j'] = vehicle_other['direction_norm_x'].values
                                df_TTC_i['hy_j'] = vehicle_other['direction_norm_y'].values
                                df_TTC_i['length_j'] = vehicle_other[model.vehicle_length_column+'_median'].values
                                df_TTC_i['width_j'] = vehicle_other[model.vehicle_width_column+'_median'].values

                                # Une os pares de conflitos no "df_TTC_t"
                                df_TTC_t = pd.concat([df_TTC_t,df_TTC_i],ignore_index=True)

                            # Calcula os conflitos nesse instante
                            df_TTC_t = TTC(df_TTC_t)
                            # Remove os "não conflitos", representado por np.inf ou com TTC = infinito
                            df_TTC_t = df_TTC_t[df_TTC_t['TTC']!=np.inf]
                            # Adicona o instante que o conflito ocorreu
                            df_TTC_t.insert(0,'frame',t)

                            # if len(df_TTC_t[df_TTC_t['TTC'].between(0,2.5)])>0:
                            #     alert_report = f' com {len(df_TTC_t[df_TTC_t["TTC"].between(0,2.5)])} conflitos detectados'
                            # else:
                            #     alert_report = ''

                            # Concatena com o dataframe geral
                            df_TTC.append(df_TTC_t)

                            # Mensagem
                            # print(f'Instante {t} processdo{alert_report}')
                        else:
                            pass
                            # print(f'Instante {t} não processdo por ter somente {len(df_analysis)} veículos')
                        print(t,"Concluído")
                    
                    df_TTC = pd.concat(df_TTC,ignore_index=True)
                    df_TTC["dv"] = np.sqrt(((df_TTC["vx_j"]-df_TTC["vx_i"])**2)+((df_TTC["vy_j"]-df_TTC["vy_i"])**2))
                    df_TTC.to_csv(save_file_path,index=False)

                    print("OK",f)
            except Exception as e:
                print("Buxo",f,e)

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
                bd_support = bd_support.merge(pd.read_excel(f"data/Dados dos vídeos consolidados.xlsx",sheet_name="Coletas")[["id_coleta","cod_motofaixa","cod_faixas"]],on="id_coleta",how="left")
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
        # root_path = "data_ignore"
        # os.chdir(root_path)

        if True:
            for f in os.listdir("data/json"):
                if not os.path.exists(f"data/collected/count_flow_speed/{f.replace('.json','.csv')}"):
                    try:
                        model = YoloMicroscopicDataProcessing()
                        model.ImportFromJSON(f"data/json/{f}",post_processing=model.PostProcessing1)

                        # traffic_lane_list = pd.read_excel(f"data/Dados dos vídeos consolidados.xlsx",sheet_name="Coletas")
                        # traffic_lane_list = traffic_lane_list[traffic_lane_list["id_coleta"].astype(str)==f.split("_")[0]]
                        # traffic_lane_list = traffic_lane_list.astype(str)["cod_faixas"].values[0].split(",")
                        # traffic_lane_list = [int(i) for i in traffic_lane_list]
                        traffic_lane_list = None

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