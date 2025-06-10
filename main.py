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

def MotorcycleBetween(id_leader,id_follower,left_instant,right_instant,side_offset_vehicle=0,frequency_check_motorcycle=1/30,summarize=False):
    # Ajuste dos limites de tempo
    left_instant = model.df.iloc[(model.df[model.instant_column]-left_instant).abs().argsort()[0]][model.instant_column]
    right_instant = model.df.iloc[(model.df[model.instant_column]-right_instant).abs().argsort()[0]][model.instant_column]

    # Todas as motocicletas que passaram na frequência verificada
    motorcycle_between = pd.DataFrame()
    # Controle de frames verificados duplamente
    frame_list = []

    for t in np.arange(left_instant,right_instant,step=frequency_check_motorcycle):
        motorcycle_t = model.VehicleAhead(id_follower,t,side_offset_vehicle=side_offset_vehicle,ignore_vehicle_types_list=model.vehicle_category_list['four_wheel']+['Pedestre'],project_verification=True)
        frame_t = int(round(t*frequency_check_motorcycle,0))

        if frame_t in frame_list:
            pass
            # print(f'O frame {frame_t} já foi verificado! Aumente o parâmetro "frequency_check_motorcycle" no valor de "{round(frequency_check_motorcycle,4)}" para um valor referente ao fps do vídeo de coleta.')
        frame_list.append(frame_t)
        motorcycle_between = pd.concat([motorcycle_between,motorcycle_t],ignore_index=True)

    # Limite do veiculo seguidor
    max_leader_frame = model.df[model.df[model.id_column]==id_leader][model.frame_column].max()
    min_leader_frame = model.df[model.df[model.id_column]==id_leader][model.frame_column].min()

    motorcycle_between['limit_distance_leader'] = motorcycle_between[model.frame_column].apply(lambda x:model.df[(model.df[model.id_column]==id_leader) & (model.df[model.frame_column]==x)][model.x_tail_column].iloc[0] if (x<=max_leader_frame) and (x>=min_leader_frame) else model.video_width)
    motorcycle_between = motorcycle_between[motorcycle_between[model.x_head_column]<=motorcycle_between['limit_distance_leader']]


    motorcycle_between = motorcycle_between.groupby(model.id_column)[model.frame_column].count().reset_index()
    motorcycle_between['time_between'] = motorcycle_between[model.frame_column]*frequency_check_motorcycle
    motorcycle_between['perc_time_between'] = motorcycle_between[model.frame_column]/len(frame_list)

    if summarize:
        abs_values_min_list = [0.25,0.5,1]
        perc_values_min_list = [25,50,75,100]

        summarize_motorcycle_between = pd.DataFrame()

        for i in abs_values_min_list:
            motorcycle_i = motorcycle_between[motorcycle_between['time_between']>=i]
            summarize_motorcycle_between[str(i)+'_sec_id'] = [motorcycle_i[model.id_column].tolist()]
            summarize_motorcycle_between[str(i)+'_sec_count'] = [len(motorcycle_i)]
        for i in perc_values_min_list:
            motorcycle_i = motorcycle_between[motorcycle_between['perc_time_between']>=i/100]
            summarize_motorcycle_between[str(i)+'%_id'] = [motorcycle_i[model.id_column].tolist()]
            summarize_motorcycle_between[str(i)+'%_count'] = [len(motorcycle_i)]

        motorcycle_between = summarize_motorcycle_between

    return motorcycle_between

def PredictLeaderID(position_dataframe,position_follower,traffic_lane):
    if position_follower==1:
        id_predict =  -1
    else:
        id_predict = position_dataframe[(position_dataframe[model.traffic_lane_column+'_follower']==traffic_lane) & (position_dataframe['position']==position_follower-1)][model.id_column+'_follower'].iloc[0]
    return id_predict

def LaneChange(id_vehicle,left_instant,right_instant,delta_lim=1):
    df_analysed = model.df[(model.df[model.id_column]==id_vehicle) & (model.df[model.instant_column].between(left_instant,right_instant))]
    delta = df_analysed['y'].max() - df_analysed['y'].min()
    print(id_vehicle,df_analysed['y'].max(),df_analysed['y'].min(),delta)
    return True if delta>delta_lim else False

def DischargeHeadwayMotorcycleAnalysis(
    left_instant,
    right_instant,
    frequency_check_motorcycle=1/10
    ):
    # Veículos durante a descarga do ciclo em seu primeio instante de aparecimento
    vehicle_cycle = model.df[model.df[model.instant_column].between(left_instant,right_instant)]
    vehicle_cycle = vehicle_cycle[vehicle_cycle[model.vehicle_type_column].isin(model.vehicle_category_list['four_wheel'])].sort_values(by=[model.traffic_lane_column,model.instant_column])
    vehicle_cycle = vehicle_cycle.groupby(model.id_column).first().reset_index()
    vehicle_cycle = vehicle_cycle[[model.id_column,model.traffic_lane_column,model.vehicle_type_column,model.instant_column]].sort_values(by=[model.traffic_lane_column,model.instant_column])

    vehicle_cycle = vehicle_cycle.rename(columns={
        model.id_column:model.id_column+'_follower',
        model.traffic_lane_column:model.traffic_lane_column+'_follower',
        model.vehicle_type_column:model.vehicle_type_column+'_follower',
        model.instant_column:model.instant_column+'_first_follower'
    })

    # Instante em que cruzou a faixa de referência
    vehicle_cycle[model.instant_column+'_crossing_follower'] = vehicle_cycle[model.id_column+'_follower'].apply(lambda x:model.InstantCrossingSection(x,section_reference=model.motobox_end_section)[0])
    vehicle_cycle = vehicle_cycle[vehicle_cycle[model.instant_column+'_crossing_follower'].between(left_instant,right_instant)]

    # Posição na descarga
    vehicle_cycle = vehicle_cycle.sort_values(by=[model.traffic_lane_column+'_follower',model.instant_column+'_crossing_follower'])
    position_list = []
    for i in vehicle_cycle[model.traffic_lane_column+'_follower'].unique():
        position_list = position_list + list(range(1,len(vehicle_cycle[vehicle_cycle[model.traffic_lane_column+'_follower']==i])+1))
    vehicle_cycle['position'] = position_list

    # Indica qual deveria ser o seguidor
    vehicle_cycle[model.id_column+'_leader_predict'] = vehicle_cycle.apply(lambda x:PredictLeaderID(vehicle_cycle,x['position'],x[model.traffic_lane_column+'_follower']),axis=1)
    # Verifica retroativamente se, quando o líder passou pela faixa de referencia, o seguidor reconhecia esse como líder
    # Útil para identificar interações mais complexas como troca e faixa
    # Isso pode ocorrer caso o headway posicional seja maior que o observável
    # Existe a lógica do líder-seguidor, mas com o headway grande, necesse caso, optou-se por não remover esses casos
    vehicle_cycle[model.id_column+'_leader'] = vehicle_cycle.apply(lambda x:model.FirstVehicleAhead(x[model.id_column+'_follower'],x[model.instant_column+'_first_follower'],ignore_vehicle_types_list=model.vehicle_category_list['two_wheel'],report_null_value=True)[model.id_column].iloc[0],axis=1)
    vehicle_cycle[model.id_column+'_leader'] = vehicle_cycle.apply(lambda x:x[model.id_column+'_leader_predict'] if (x[model.id_column+'_leader']==-1) and (x[model.id_column+'_leader_predict']!=-1) else x[model.id_column+'_leader'],axis=1)

    # Reserva os primeiros veículos por ter uma lógica diferente de cálculo
    first_vehicle_cycle = vehicle_cycle[vehicle_cycle['position']==1]
    # Remover veículos sem líder
    vehicle_cycle = vehicle_cycle[vehicle_cycle[model.id_column+'_leader']!=-1]
    # Instante em que o líder cruza a faixa
    vehicle_cycle[model.instant_column+'_crossing_leader'] = vehicle_cycle[model.id_column+'_leader'].apply(lambda x:model.InstantCrossingSection(x,section_reference=model.motobox_end_section)[0])
    # Ajustes nos primeiro veículos
    first_vehicle_cycle[model.instant_column+'_crossing_leader'] = [left_instant]*len(first_vehicle_cycle)
    first_vehicle_cycle['impacted_lane_change'] = [False]*len(first_vehicle_cycle)

    # Verificação de troca de faixa
    # LC1 - Veículo seguidor troca de faixa
    vehicle_cycle['LC1'] = vehicle_cycle.apply(lambda x:LaneChange(x[model.id_column+'_follower'],x[model.instant_column+'_first_follower'],x[model.instant_column+'_crossing_follower']),axis=1)
    # LC2 - Verifica se o veículo atrás do lider no momento em que o líder cruza a faixa de retenção é diferente ou não do seguidor
    # vehicle_cycle['LC2'] = vehicle_cycle.apply(lambda x:FirstVehicleBehind2(x[df_YMDP.id_column+'_leader'],x[df_YMDP.instant_column+'_crossing_leader'],ignore_vehicle_types_list=df_YMDP.vehicle_category_list['two_wheel']),axis=1)!=vehicle_cycle[df_YMDP.id_column+'_follower']
    vehicle_cycle['LC2'] = vehicle_cycle.apply(lambda x:model.FirstVehicleAhead(x[model.id_column+'_follower'],x[model.instant_column+'_crossing_leader'],ignore_vehicle_types_list=model.vehicle_category_list['two_wheel'],project_verification=True,report_null_value=True)[model.id_column].iloc[0],axis=1)!=vehicle_cycle[model.id_column+'_leader']
    # LC3 - Se o seguidor mudou de faixa (LC1 = True), ele não pode ser líder
    vehicle_cycle['LC3'] = vehicle_cycle.apply(lambda x:x[model.id_column+'_leader'] in vehicle_cycle[vehicle_cycle['LC1']==True][model.id_column+'_follower'].tolist(),axis=1)
    # LC4 - Se o seguidor trocou de faixa, o líder dele não pode ser mais líder
    vehicle_cycle['LC4'] = vehicle_cycle.apply(lambda x:x[model.id_column+'_leader'] in vehicle_cycle[vehicle_cycle['LC1']==True][model.id_column+'_leader'].tolist(),axis=1)
    vehicle_cycle['impacted_lane_change'] = vehicle_cycle[['LC1','LC2','LC3','LC4']].sum(axis=1)>0

    # Unir dados da primeira posição
    vehicle_cycle = pd.concat([vehicle_cycle,first_vehicle_cycle],ignore_index=True).sort_values(by=[model.traffic_lane_column+'_follower','position'])
    # Remover colunas intermediárias
    vehicle_cycle = vehicle_cycle.drop(columns=['LC1','LC2','LC3','LC4',model.id_column+'_leader_predict'])

    # Cálculo do headway de descarga
    vehicle_cycle['headway'] = vehicle_cycle['instant_crossing_follower'] - vehicle_cycle['instant_crossing_leader']

    # Cálculo das motocicletas
    motorcylce_classes = pd.DataFrame()
    for i in range(len(vehicle_cycle)):
        try:
            vehicle_cycle_i = vehicle_cycle.iloc[i]
            print(vehicle_cycle_i[model.id_column+'_leader'],vehicle_cycle_i[model.id_column+'_follower'],'Início')
            motorcylce_classes_i = MotorcycleBetween(vehicle_cycle_i[model.id_column+'_leader'],vehicle_cycle_i[model.id_column+'_follower'],vehicle_cycle_i[model.instant_column+'_crossing_leader'],vehicle_cycle_i[model.instant_column+'_crossing_follower'],frequency_check_motorcycle=frequency_check_motorcycle,summarize=False,side_offset_vehicle=0.3)
            motorcylce_classes_j = MotorcycleBetween(vehicle_cycle_i[model.id_column+'_leader'],vehicle_cycle_i[model.id_column+'_follower'],vehicle_cycle_i[model.instant_column+'_crossing_leader'],vehicle_cycle_i[model.instant_column+'_crossing_follower'],frequency_check_motorcycle=frequency_check_motorcycle,summarize=False,side_offset_vehicle=1)

            j_i = motorcylce_classes_j.merge(motorcylce_classes_i,on='id',how='left',suffixes=('_all_range','_between')).fillna(0)
            j_i['perc_time_virtual_lane'] = j_i['perc_time_between_all_range']-j_i['perc_time_between_between']

            id_virtual_lane = j_i[(j_i['perc_time_between_all_range']>=0.5) & (j_i['perc_time_virtual_lane']>j_i['perc_time_between_between'])][model.id_column].tolist()
            id_between = j_i[(j_i['perc_time_between_all_range']>=0.5) & (j_i['perc_time_between_between']>=j_i['perc_time_virtual_lane'])][model.id_column].tolist()

            motorcylce_classes_summarize = pd.DataFrame()
            motorcylce_classes_summarize[model.id_column+'_follower'] = [vehicle_cycle_i[model.id_column+'_follower']]
            motorcylce_classes_summarize[model.id_column+'_motorcycle_virtual_lane'] = [id_virtual_lane]
            motorcylce_classes_summarize['count_motorcycle_virtual_lane'] = [len(id_virtual_lane)]
            motorcylce_classes_summarize[model.id_column+'_motorcycle_between'] = [id_between]
            motorcylce_classes_summarize['count_motorcycle_between'] = [len(id_between)]

            motorcylce_classes = pd.concat([motorcylce_classes,motorcylce_classes_summarize],ignore_index=True)
            print(vehicle_cycle_i[model.id_column+'_leader'],vehicle_cycle_i[model.id_column+'_follower'],'OK')

        except Exception as e:
            print(e,'Não OK')

    vehicle_cycle = vehicle_cycle.merge(motorcylce_classes,on=model.id_column+'_follower',how='left')

    return vehicle_cycle

if __name__=="__main__":
    mode = "test"
    
    if mode=="test":
        model = YoloMicroscopicDataProcessing()
        model.ImportFromJSON("data/json/C_x_13M_SemMotobox_D5_0001.json",post_processing=model.PostProcessing1)

        start_instant = model.green_open_time[0]
        last_instant = model.green_open_time[1]

        result = model.DischargeHeadwayMotorcycleAnalysis(
            start_frame=int(model.fps*start_instant),
            last_frame=int(model.fps*last_instant)
        )

        print(result)

        # hd = pd.DataFrame()

        # for i in range(len(model.green_open_time)-1):
        #     print(model.green_open_time[i],model.green_open_time[i+1], 'Inicio')
        #     hd_i = DischargeHeadwayMotorcycleAnalysis(model.green_open_time[i],model.green_open_time[i+1],frequency_check_motorcycle=1/10)
        #     hd = pd.concat([hd,hd_i],ignore_index=True)
        #     print(model.green_open_time[i],model.green_open_time[i+1], 'Fim')

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