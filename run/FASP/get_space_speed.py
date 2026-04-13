import sys
import os
sys.path.append(os.path.abspath(".")) 

from model import *
import timeit
import warnings
warnings.filterwarnings('ignore')

def GetTrafficLaneFromSupport(file_path,file_name):
    bd_video = pd.read_excel(file_path,sheet_name="Vídeos")
    bd_coleta = pd.read_excel(file_path,sheet_name="Coletas")

    bd_video = bd_video[bd_video["id_video"]==file_name.split(".")[0]]
    bd_support = bd_video.merge(bd_coleta,on="id_coleta",how="left")
    traffic_lane_list = [int(i) for i in bd_support.astype(str)["cod_faixas"].values[0].split(",")]

    return traffic_lane_list

if __name__=="__main__":
    root_path = "project/Faixa Azul SP"
    os.chdir(root_path)

    # Início do loop
    for f in os.listdir("data/json"):
        output_file_path = os.path.join("data/collected/Dissertação/MotorcycleSpaceSpeed",f.replace(".json",".csv"))
        if not os.path.exists(output_file_path):
            # File traj
            model = YoloMicroscopicDataProcessing()
            model.ImportFromJSON(os.path.join("data/json",f),post_processing=model.PostProcessing1)
            f = f.replace(".json",".csv")
            # print(model.df.head())

            # Traffic lanes
            traffic_lane_list = GetTrafficLaneFromSupport(
                "./Dados dos vídeos consolidados.xlsx",
                f
            )
            model.df = model.df[model.df["traffic_lane"].isin(traffic_lane_list)]

            # Planilha em que cada linha é uma moto e seus confinantes
            df_motorcycle = model.df[model.df[model.vehicle_type_column].isin(['Moto'])]# .iloc[:100]
            df_motorcycle = df_motorcycle.apply(lambda row:model.FR2SMapping(row["id"],row["frame"]).iloc[0],axis=1,result_type="expand")
            df_motorcycle[[
                "id",
                "frame",
                "front",
                "rear",
                "left",
                "right"]] = df_motorcycle[[
                "id",
                "frame",
                "front",
                "rear",
                "left",
                "right"]].astype(int)

            # Front gap
            df_motorcycle["front_gap"] = df_motorcycle["dist_front"]

            # Data from id
            df_vehicle = df_motorcycle[["id","frame"]]
            df_vehicle = df_vehicle.merge(
                model.df[[
                    "id",
                    "frame",
                    "vehicle_type",
                    "head",
                    "vehicle_length",
                    "x_instant_speed",
                    "y_instant_speed",
                    "instant_speed"
                ]],
                on=["id","frame"],
                how="left")
            df_vehicle.columns = [col+"_reference" if not col in ["id","frame"] else col for col in df_vehicle.columns]
            df_motorcycle = df_motorcycle.merge(df_vehicle,on=["id","frame"],how="left")

            # Front space headway
            df_front_vehicle = df_motorcycle[["frame","front"]].copy().rename(columns={"front":"id"})
            df_front_vehicle = df_front_vehicle.merge(
                model.df[[
                    "id",
                    "frame",
                    "vehicle_type",
                    "head",
                    "vehicle_length",
                    "x_instant_speed",
                    "y_instant_speed",
                    "instant_speed"
                ]],
                on=["frame","id"],
                how="left")
            df_front_vehicle.columns = [col+"_front" if not col in ["id","frame"] else col for col in df_front_vehicle.columns]
            df_front_vehicle = df_front_vehicle.rename(columns={"id":"front"})
            df_motorcycle = df_motorcycle.merge(df_front_vehicle,on=["front","frame"],how="left")

            # Headway
            df_motorcycle["space_headway"] = df_motorcycle["head_front"] - df_motorcycle["head_reference"]
            # Delta Vx
            df_motorcycle["delta_instant_speed_x"] =  df_motorcycle["x_instant_speed_reference"] - df_motorcycle["x_instant_speed_front"]
            # TTC Long
            df_motorcycle["TTC"] = df_motorcycle["front_gap"] / df_motorcycle["delta_instant_speed_x"]

            df_motorcycle.to_csv(output_file_path,index=False)
            print("OK",f)
        else:
            print("Já processado",f)
    

