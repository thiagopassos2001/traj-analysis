import numpy as np
import pandas as pd
from model import *
import timeit
import os
import warnings
import shapely
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from until.SSMsOnPlane.src.longitudinal_ssms import TTC
from until.SSMsOnPlane.src.two_dimensional_ssms import TTC2D
from until.SSMsOnPlane.src.efficiency_utils import evaluate_efficiency

# samples = pd.read_hdf('until/SSMsOnPlane/assets/samples.h5', key='example')
# SSMs = ['TTC', 'DRAC', 'MTTC', 'PSD', 'TTC2D', 'ACT', 'TAdv']

# results = TTC(samples, toreturn='dataframe')
# print(results.head())

root_path = "project/Safe Lane"
file_name = "DJI_0001_transformed_processed.json"
df_concat = []
start_timer = timeit.default_timer()
if __name__=="__main__":
    os.chdir(root_path)

    for file_name in [
        "DJI_0001_transformed_processed.json",
    ]:
        model = YoloMicroscopicDataProcessing()
        model.ImportFromJSON2(os.path.join("data/json",file_name))

        # Cálculo nas inclinações e dimensões "teóricas" dos veículos
        df_ = []
        for i in model.df["id"].unique():
            print(i)
            df_.append(model.DirectionEstimate(i,window_length=15))
        df_ = pd.concat(df_,ignore_index=True).sort_values(by=["frame","id"])

        # Cálculo do TTC
        df_TTC = []
        for f in model.df["frame"].unique():
            df_f = df_[df_["frame"]==f]

            # Ajuste das coordenadas y
            df_f["y"] = model.video_heigth - df_f["y"]
            df_f["y_instant_speed"] = -df_f["y_instant_speed"]
            df_f['direction_norm_y'] = -df_f['direction_norm_y']

            if len(df_f)>1:
                # Pares de conflitos no instante t
                df_TTC_t = pd.DataFrame(columns=['id_i'])

                for id_vehicle_i in df_f["id"]:
                    # Veículo "i" do conflito
                    vehicle_i = df_f[df_f["id"]==id_vehicle_i]
                    # Demais veículos "j" que podem estar em conflito com o veículo de referencia
                    vehicle_other = df_f[df_f["id"]!=id_vehicle_i]

                    # Evitar o cáluclo de pares duplicados
                    # Se o veículo "j" nesse instante já tiver sido computado como "i" em
                    # Alguma iteração anterior, esse conflito potencial já foi computado
                    # Portanto esse veículo não é mais habilitado para ser "j" pois já foi "i"
                    vehicle_other = vehicle_other[-vehicle_other["id"].isin(df_TTC_t['id_i'])]

                    # Dados do veículo "i"
                    df_TTC_i = pd.DataFrame()
                    df_TTC_i['track_id_i'] = [id_vehicle_i]*len(vehicle_other)
                    df_TTC_i['vehicle_type_i'] = vehicle_i[model.vehicle_type_column].iloc[0]
                    df_TTC_i['time'] = f
                    df_TTC_i['x_i'] = vehicle_i[model.x_centroid_column].iloc[0]
                    df_TTC_i['y_i'] = vehicle_i[model.y_centroid_column].iloc[0]
                    df_TTC_i['vx_i'] = vehicle_i[model.x_instant_speed_column].iloc[0]
                    df_TTC_i['vy_i'] = vehicle_i[model.y_instant_speed_column].iloc[0]
                    df_TTC_i['hx_i'] = vehicle_i['direction_norm_x'].iloc[0]
                    df_TTC_i['hy_i'] = vehicle_i['direction_norm_y'].iloc[0]
                    df_TTC_i['length_i'] = vehicle_i["theoretical_length"].iloc[0]
                    df_TTC_i['width_i'] = vehicle_i["theoretical_width"].iloc[0]
                    df_TTC_i["acc_i"] = vehicle_i[model.instant_acc_column].iloc[0]

                    # Dados dos veículos "j"
                    df_TTC_i['track_id_j'] = vehicle_other[model.id_column].values
                    df_TTC_i['vehicle_type_j'] = vehicle_other[model.vehicle_type_column].values
                    df_TTC_i['x_j'] = vehicle_other[model.x_centroid_column].values
                    df_TTC_i['y_j'] = vehicle_other[model.y_centroid_column].values
                    df_TTC_i['vx_j'] = vehicle_other[model.x_instant_speed_column].values
                    df_TTC_i['vy_j'] = vehicle_other[model.y_instant_speed_column].values
                    df_TTC_i['hx_j'] = vehicle_other['direction_norm_x'].values
                    df_TTC_i['hy_j'] = vehicle_other['direction_norm_y'].values
                    df_TTC_i['length_j'] = vehicle_other["theoretical_length"].values
                    df_TTC_i['width_j'] = vehicle_other["theoretical_width"].values
                    df_TTC_i["acc_j"] = vehicle_other[model.instant_acc_column].values

                    # Une os pares de conflitos no "df_TTC_t"
                    df_TTC_t = pd.concat([df_TTC_t,df_TTC_i],ignore_index=True)

                # Calcula os conflitos nesse instante
                df_TTC_t = TTC(df_TTC_t, toreturn='dataframe')
                # Remove os "não conflitos", representado por np.inf ou com TTC = infinito
                # df_TTC_t = df_TTC_t[df_TTC_t['TTC']!=np.inf]

                # Concatena com o dataframe geral
                df_TTC.append(df_TTC_t)

                # Mensagem
                print(f'Instante {f} processdo com {len(df_TTC_t[(df_TTC_t['TTC']!=np.inf) & (df_TTC_t['TTC']<=1.5)])} conflitos relevantes')
            else:
                print(f'Instante {f} não processdo por ter somente {len(df_f)} veículos')

        df_TTC = pd.concat(df_TTC,ignore_index=True)
        df_TTC.to_csv("TTC_test.csv",index=False)
