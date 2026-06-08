import sys
sys.path.append(r"C:\Users\thiag\OneDrive\Documents\Repositórios (Local)\traj-analysis") 
import matplotlib.pyplot as plt
import seaborn as sns
from model import *
import pandas as pd
import os
import warnings
from shapely.geometry import Point
warnings.filterwarnings('ignore')
from scipy.signal import savgol_filter
import timeit

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
    
    start_timer = timeit.default_timer()
    root_path = "project/Faixa Azul (Fortaleza)"

    os.chdir(root_path)

    df_support = GetVideoDataset(agg_var=["id_video","nome_video"])
    id_coleta_list = ["2025-08-06","2026-05-21"]
    df_support = df_support[df_support["id_coleta"].isin(id_coleta_list)]
    df_support["id"] = df_support["id"]+"_processed.json"

    # Início do loop
    for f in os.listdir("data/json"):
        save_file_path = f"data/collected/MotorcycleRoadSection/{f.replace('.json','.csv')}"
        if not os.path.exists(save_file_path):
            model = YoloMicroscopicDataProcessing()
            model.ImportFromJSON2(f"data/json/{f}") 
            
            df_support_ = df_support[df_support["id"]==f]
            traffic_region_list = eval(df_support_.iloc[0]["cod_faixas"])
            # cod_motorcycle_lane = df_support_.iloc[0]["cod_motofaixa"]

            if df_support_["cod_motofaixa"].isna().iloc[0]:
                req_traffic_regions = [df_support_.iloc[0]["faixa_1"],df_support_.iloc[0]["faixa_2"]]
            else:
                req_traffic_regions = [df_support_.iloc[0]["cod_motofaixa"]]
            
            model.df = model.df[model.df["traffic_region"].isin(traffic_region_list)]
            df_motorcycle = model.df[model.df["vehicle_type"]=="Moto"]

            # x1y1
            df_motorcycle = gpd.GeoDataFrame(df_motorcycle,geometry=gpd.points_from_xy(df_motorcycle["p1xbb"],df_motorcycle["p1ybb"]),crs="EPSG:31984")
            df_motorcycle = gpd.sjoin(df_motorcycle,model.regions.rename(columns={"id":"x1y1_tr"})[["x1y1_tr","geometry"]],how="left",predicate="within",).drop(columns=["index_right"])
            df_motorcycle = df_motorcycle.drop_duplicates(subset=["frame","id"])

            # x1y2
            df_motorcycle = gpd.GeoDataFrame(df_motorcycle,geometry=gpd.points_from_xy(df_motorcycle["p1xbb"],df_motorcycle["p2ybb"]),crs="EPSG:31984")
            df_motorcycle = gpd.sjoin(df_motorcycle,model.regions.rename(columns={"id":"x1y2_tr"})[["x1y2_tr","geometry"]],how="left",predicate="within",).drop(columns=["index_right"])
            df_motorcycle = df_motorcycle.drop_duplicates(subset=["frame","id"])

            # x2y1
            df_motorcycle = gpd.GeoDataFrame(df_motorcycle,geometry=gpd.points_from_xy(df_motorcycle["p2xbb"],df_motorcycle["p1ybb"]),crs="EPSG:31984")
            df_motorcycle = gpd.sjoin(df_motorcycle,model.regions.rename(columns={"id":"x2y1_tr"})[["x2y1_tr","geometry"]],how="left",predicate="within",).drop(columns=["index_right"])
            df_motorcycle = df_motorcycle.drop_duplicates(subset=["frame","id"])

            # x2y2
            df_motorcycle = gpd.GeoDataFrame(df_motorcycle,geometry=gpd.points_from_xy(df_motorcycle["p2xbb"],df_motorcycle["p2ybb"]),crs="EPSG:31984")
            df_motorcycle = gpd.sjoin(df_motorcycle,model.regions.rename(columns={"id":"x2y2_tr"})[["x2y2_tr","geometry"]],how="left",predicate="within",).drop(columns=["index_right"])
            df_motorcycle = df_motorcycle.drop_duplicates(subset=["frame","id"])

            # checks
            # motofaixa
            df_motorcycle["motorcycle_check"] = (df_motorcycle[["x1y1_tr","x1y2_tr","x2y1_tr","x2y2_tr"]].fillna("").apply(lambda x: "_".join(v for v in x if v != ""), axis=1))
            df_motorcycle["motorcycle_check"] = df_motorcycle["motorcycle_check"].apply(lambda x: all(item in x.split("_") for item in req_traffic_regions))

            # outros corredores
            df_motorcycle["other_corridor_check"] = (df_motorcycle[["x1y1_tr","x1y2_tr","x2y1_tr","x2y2_tr"]].fillna("").nunique(axis=1).ge(2)) & -df_motorcycle["motorcycle_check"]

            virtual_lane = {
                -1:"Fora do Corredor",
                1:"Corredor Principal",
                0:"Outros Corredores"
            }

            df_motorcycle["virtual_lane"] = df_motorcycle.apply(lambda row:1 if row["motorcycle_check"] else (0 if row["other_corridor_check"] else -1),axis=1)
            df_motorcycle["virtual_lane_type"] = df_motorcycle["virtual_lane"].apply(lambda x:virtual_lane[x])
            df_motorcycle = df_motorcycle[["frame","id","virtual_lane","virtual_lane_type","x1y1_tr","x1y2_tr","x2y1_tr","x2y2_tr","motorcycle_check","other_corridor_check"]]
            df_motorcycle.to_csv(save_file_path,index=False)
            print(f"{f} processado!")

        else:
            print(f"{f} já processado!")