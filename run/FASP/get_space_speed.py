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

def GetVideoDataset(agg_var=["id_coleta","nome_video"]):

    sheet_url_video = "https://docs.google.com/spreadsheets/d/1pN1dvag90MUecKgGvSCfjpun5Ym3Yl-oNG7PH1TQ8gE/export?format=csv&gid=2080708038"
    df_video = pd.read_csv(sheet_url_video).reset_index(drop=True)
    df_video[["id_video","id_coleta"]] = df_video[["id_video","id_coleta"]].astype(str)

    sheet_url_img_ref = "https://docs.google.com/spreadsheets/d/1pN1dvag90MUecKgGvSCfjpun5Ym3Yl-oNG7PH1TQ8gE/export?format=csv&gid=1167794226"
    df_img_ref = pd.read_csv(sheet_url_img_ref).reset_index(drop=True)
    df_img_ref[["id_video","id_coleta"]] = df_img_ref[["id_video","id_coleta"]].astype(str)

    df = df_video.merge(df_img_ref,on=["id_coleta","img_ref"],how="left",suffixes=["","_img_ref"])

    # Create id
    df.insert(0,"id",df[agg_var].agg('_'.join, axis=1))

    return df

if __name__=="__main__":
    # root_path = "project/Faixa Azul SP"
    root_path = "project/Faixa Azul (Fortaleza)"
    os.chdir(root_path)





    
    # FA Fortaleza
    df_support = GetVideoDataset(agg_var=["id_video","nome_video"])
    # id_coleta_list = ["2025-08-06","2026-05-21"]
    id_coleta_list = ["2025-08-07"]
    df_support = df_support[df_support["id_coleta"].isin(id_coleta_list)]
    df_support["id"] = df_support["id"]+"_processed.json"






    # Início do loop
    for f in os.listdir("data/json"):
        output_file_path = os.path.join("data/collected/MotorcycleSpaceSpeed",f.replace(".json",".csv")) # /Dissertação
        if not os.path.exists(output_file_path):


            # FA Fortaleza
            df_support_ = df_support[df_support["id"]==f]
            traffic_lane_list = eval(df_support_.iloc[0]["cod_faixas"])


            # File traj
            model = YoloMicroscopicDataProcessing()
            model.ImportFromJSON2(os.path.join("data/json",f)) # ImportFromJSON2 ,post_processing=model.PostProcessing1
            model.df = model.df.rename(columns={"traffic_region":"traffic_lane"}) # FA Fortaleza
            f = f.replace(".json",".csv")
            # print(model.df.head())

            # # Traffic lanes (FASP)
            # traffic_lane_list = GetTrafficLaneFromSupport(
            #     "./Dados dos vídeos consolidados.xlsx",
            #     f
            # )

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
                    "x","y",
                    "head",
                    "vehicle_length",
                    "vehicle_width",
                    "x_instant_speed",
                    "y_instant_speed",
                    "x_instant_acc"
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
                    "x","y",
                    "head",
                    "vehicle_length",
                    "vehicle_width",
                    "x_instant_speed",
                    "y_instant_speed",
                    "x_instant_acc"
                ]],
                on=["frame","id"],
                how="left")
            df_front_vehicle.columns = [col+"_front" if not col in ["id","frame"] else col for col in df_front_vehicle.columns]
            df_front_vehicle = df_front_vehicle.rename(columns={"id":"front"})
            df_motorcycle = df_motorcycle.merge(df_front_vehicle,on=["front","frame"],how="left")

            # Max obs space_headway
            df_motorcycle["max_obs_space_headway"] = model.video_width
            # Headway
            df_motorcycle["space_headway"] = df_motorcycle["head_front"] - df_motorcycle["head_reference"]
            # Delta Vx
            df_motorcycle["delta_instant_speed_x"] =  df_motorcycle["x_instant_speed_reference"] - df_motorcycle["x_instant_speed_front"]
            # Degree aligment
            df_motorcycle["degree_aligment"] = np.rad2deg(np.arctan((df_motorcycle["y_front"]-df_motorcycle["y_reference"])/(df_motorcycle["x_front"]-df_motorcycle["x_reference"])))
            # TTC Long
            df_motorcycle["TTC"] = df_motorcycle["front_gap"] / df_motorcycle["delta_instant_speed_x"]

            df_motorcycle.to_csv(output_file_path,index=False)
            print("OK",f)
        else:
            print("Já processado",f)
    

