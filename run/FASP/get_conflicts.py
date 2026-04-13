import sys
import os
sys.path.append(os.path.abspath(".")) 

# from model import *
import timeit
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

if __name__=="__main__":
    root_path = r"C:\Users\thiag\OneDrive\Documents\Repositórios (Local)\traj-analysis\project\Faixa Azul SP"
    os.chdir(root_path)
    buffer_frame_mean = 7

    # Início do loop
    for f in os.listdir("data/collected/Dissertação/MotorcycleSpaceSpeed"):
        # File motorcycle space speed
        df_mss = pd.read_csv(os.path.join("data/collected/Dissertação/MotorcycleSpaceSpeed",f))
        # File motorcycle road section
        df_mrs_OG = pd.read_csv(os.path.join("data/collected/Dissertação/MotorcycleRoadSection",f))

        df_mrs = df_mrs_OG[df_mrs_OG["virtual_lane_type"]=="Corredor Principal"].copy()

        # Filter 1
        # df_mrs = df_mrs[df_mrs["virtual_lane_type"]=="Corredor Principal"] # commented by run all
        # Merge
        df_mrs = df_mrs.merge(df_mss,on=["id","frame"],how="left")
        # Calc prioritize (Y)
        df_mrs = df_mrs.groupby(["id"]).agg({ # ,"virtual_lane_type" # commented by run all
            "frame":"count",
            "x_instant_speed_reference":"max",
            "space_headway":"min",
            "TTC":"min"
            }).reset_index(drop=False)
        # Filter 2
        df_mrs = df_mrs.rename(columns={
            "frame":"frame_count",
            # "x_instant_speed_reference":"m1_speed",
            # "space_headway":"m2_headway",
            # "TTC":"m3_TTC"
            })
        df_mrs = df_mrs[df_mrs["frame_count"]>=20] # commented by run all default -> 45
        
        # Get Xs (remerge)
        # m1
        df_m1 = df_mrs[["id","frame_count","x_instant_speed_reference"]].copy()
        df_m1 = df_m1.merge(df_mss[[
            "id",
            "x_instant_speed_reference",
            "frame",
            "vehicle_type_reference",
            "front",
            "vehicle_type_front",
            "front_gap",
            "space_headway"
        ]],on=["id","x_instant_speed_reference"],how="left")
        df_m1 = df_m1.drop_duplicates(subset=["id","x_instant_speed_reference"])
        # Get virtual_lane_type
        df_m1 = df_m1.merge(df_mrs_OG,on=["id","frame"],how="left")
        # Mean parameters
        for p in ["x_instant_speed_reference","front_gap","space_headway"]:
            df_m1[p+"_mean"] = df_m1.apply(lambda row:df_mss[(df_mss["id"]==row["id"]) & (df_mss["frame"].between(row["frame"]-buffer_frame_mean,row["frame"]+buffer_frame_mean))][p].mean(),axis=1)

        # m2
        df_m2 = df_mrs[["id","frame_count","space_headway"]].merge(df_mss[[
            "id",
            "space_headway",
            "frame",
            "vehicle_type_reference",
            "front",
            "vehicle_type_front",
            "front_gap",
            "x_instant_speed_reference"
        ]],on=["id","space_headway"],how="left")
        df_m2 = df_m2.drop_duplicates(subset=["id","space_headway"])
        # Get virtual_lane_type
        df_m2 = df_m2.merge(df_mrs_OG,on=["id","frame"],how="left")
        # Mean parameters
        for p in ["space_headway","front_gap","x_instant_speed_reference"]:
            df_m2[p+"_mean"] = df_m2.apply(lambda row:df_mss[(df_mss["id"]==row["id"]) & (df_mss["frame"].between(row["frame"]-buffer_frame_mean,row["frame"]+buffer_frame_mean))][p].mean(),axis=1)

        # m3
        df_m3 = df_mrs[["id","frame_count","TTC"]].merge(df_mss[[
            "id",
            "TTC",
            "frame",
            "vehicle_type_reference",
            "front",
            "vehicle_type_front",
            "front_gap",
            "x_instant_speed_reference",
            "x_instant_speed_front",
            "delta_instant_speed_x",
        ]],on=["id","TTC"],how="left")
        df_m3 = df_m3.drop_duplicates(subset=["id","TTC"])
        # Get virtual_lane_type
        df_m3 = df_m3.merge(df_mrs_OG,on=["id","frame"],how="left")
        # Mean parameters
        for p in ["TTC","front_gap","x_instant_speed_reference","x_instant_speed_front","delta_instant_speed_x"]:
            df_m3[p+"_mean"] = df_m3.apply(lambda row:df_mss[(df_mss["id"]==row["id"]) & (df_mss["frame"].between(row["frame"]-buffer_frame_mean,row["frame"]+buffer_frame_mean))][p].mean(),axis=1)

        # renames
        df_m1.columns = [col+"_m1" if not col in ["id","frame_count","vehicle_type_reference","virutal_lane","virtual_lane_type"] else col for col in df_m1.columns]
        df_m2.columns = [col+"_m2" if not col in ["id","frame_count","vehicle_type_reference","virutal_lane","virtual_lane_type"] else col for col in df_m2.columns]
        df_m3.columns = [col+"_m3" if not col in ["id","frame_count","vehicle_type_reference","virutal_lane","virtual_lane_type"] else col for col in df_m3.columns]
        
        # df_concat = df_m1.merge(df_m2,on=["id","frame_count","vehicle_type_reference"],how="left").merge(df_m3,on=["id"],how="left")
        # df_concat.to_excel("Teste.xlsx",index=False)
        
        df_m1.to_csv(os.path.join("data/collected/Dissertação/Interaction/All Lanes/m1",f.replace(".json",".csv")),index=False)
        df_m2.to_csv(os.path.join("data/collected/Dissertação/Interaction/All Lanes/m2",f.replace(".json",".csv")),index=False)
        df_m3.to_csv(os.path.join("data/collected/Dissertação/Interaction/All Lanes/m3",f.replace(".json",".csv")),index=False)

        print("OK",f)

    