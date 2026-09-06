import sys
import os
sys.path.append(os.path.abspath(".")) 

# from model import *
import timeit
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

def FilterMainInteraction(df):
    df_filtered = []
    df_copy = df.copy()

    id_list = df_copy["id"].unique().tolist()
    for idx in id_list:
        df_ = df_copy[df_copy["id"]==idx]
        id_front = df_[df_["front"]>0]["front"].unique().tolist()
        
        if len(id_front)>0:
            df_filtered.append(df_[df_["front"]==id_front[0]])
        else:
            df_filtered.append(df_)

    df_filtered = pd.concat(df_filtered,ignore_index=True)

    return df_filtered

if __name__=="__main__":
    root_path = r"C:\Users\thiag\OneDrive\Documents\Repositórios (Local)\traj-analysis\project\Faixa Azul SP"
    os.chdir(root_path)
    buffer_frame_mean = 7

    get_vars = [
        "id",
        "frame",
        "max_obs_space_headway",

        "vehicle_type_reference",
        "x_instant_speed_reference",
        "y_instant_speed_reference",
        "x_instant_acc_reference",

        "front",
        "vehicle_type_front",
        "x_instant_speed_front",
        "y_instant_speed_front",
        "x_instant_acc_front",

        "delta_instant_speed_x",
        "degree_aligment",
        "front_gap",
        "space_headway",
        "TTC",

        "right",
        "dist_right",
        "left",
        "dist_left",
        "real_width_lane"
    ]

    avg_vars = [
        "x_instant_speed_reference",
        "x_instant_speed_front",
        "delta_instant_speed_x",
        "degree_aligment",
        "front_gap",
        "space_headway",
        "TTC",
        "right",
        "dist_right",
        "left",
        "dist_left",
        "real_width_lane"
    ]

    # Início do loop
    for f in os.listdir("data/collected/Dissertação/MotorcycleSpaceSpeed"):
        # File motorcycle space speed
        df_mss = pd.read_csv(os.path.join("data/collected/Dissertação/MotorcycleSpaceSpeed",f))

        # for col in df_mss.columns:
        #     if df_mss[col].dtype == "int64":
        #         df_mss[col] = pd.to_numeric(df_mss[col], downcast="integer")
        #     if df_mss[col].dtype == "float64":
        #         df_mss[col] = pd.to_numeric(df_mss[col], downcast="float")


        # File motorcycle road section
        df_mrs_OG = pd.read_csv(os.path.join("data/collected/Dissertação/MotorcycleRoadSection",f))
        # for col in df_mrs_OG.columns:
        #     if df_mrs_OG[col].dtype == "int64":
        #         df_mrs_OG[col] = pd.to_numeric(df_mrs_OG[col], downcast="integer")
        #     if df_mrs_OG[col].dtype == "float64":
        #         df_mrs_OG[col] = pd.to_numeric(df_mrs_OG[col], downcast="float")

        df_mrs = df_mrs_OG[df_mrs_OG["virtual_lane_type"]=="Corredor Principal"].copy() # 

        # Filter start and and
        max_obs_space_headway = max(df_mss["max_obs_space_headway"])
        df_mss = df_mss[df_mss["head_reference"].between(3,max_obs_space_headway-3)]
        df_mss = df_mss[df_mss["head_front"].isna() | df_mss["head_front"].between(3,max_obs_space_headway-3)]
        df_mss[["right","left"]] = df_mss[["right","left"]]>0
        df_mss[["right","left"]] = df_mss[["right","left"]].astype(int)
        df_mss["real_width_lane"] = df_mss["right"] * df_mss["left"] * (df_mss["dist_left"]+df_mss["vehicle_width_reference"]+df_mss["dist_right"])

        # Filtred df for m2 and m4
        df_mss_filtered = FilterMainInteraction(df_mss)

        # Filter 1
        # df_mrs = df_mrs[df_mrs["virtual_lane_type"]=="Corredor Principal"] # commented by run all
        # Merge
        df_mrs = df_mrs.merge(df_mss,on=["id","frame"],how="left")
        # Calc prioritize (Y)
        df_mrs = df_mrs.groupby(["id","virtual_lane_type"]).agg({ # ,"virtual_lane_type" # commented by run all
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
        df_mrs = df_mrs[df_mrs["frame_count"]>=45] # commented by run all default -> 45
        
        # Get Xs (remerge)
        # m1
        df_m1 = df_mrs[["id","frame_count","x_instant_speed_reference"]].merge(df_mss[get_vars],on=["id","x_instant_speed_reference"],how="left")
        df_m1 = df_m1.drop_duplicates(subset=["id","x_instant_speed_reference"])
        # Get virtual_lane_type
        df_m1 = df_m1.merge(df_mrs_OG,on=["id","frame"],how="left")
        # Mean parameters
        for p in avg_vars:
            df_m1[p+"_mean"] = df_m1.apply(lambda row:df_mss[(df_mss["id"]==row["id"]) & (df_mss["frame"].between(row["frame"]-buffer_frame_mean,row["frame"]+buffer_frame_mean))][p].mean(),axis=1)

        # m2
        # df_m2 = df_mrs[["id","frame_count","space_headway"]].merge(df_mss_filtered[get_vars],on=["id","space_headway"],how="left")
        df_m2 = df_mrs[["id","frame_count","space_headway"]].merge(df_mss[get_vars],on=["id","space_headway"],how="left")
        df_m2 = df_m2.drop_duplicates(subset=["id","space_headway"])
        # Get virtual_lane_type
        df_m2 = df_m2.merge(df_mrs_OG,on=["id","frame"],how="left")
        # Mean parameters
        for p in avg_vars:
            # df_m2[p+"_mean"] = df_m2.apply(lambda row:df_mss_filtered[(df_mss_filtered["id"]==row["id"]) & (df_mss_filtered["frame"].between(row["frame"]-buffer_frame_mean,row["frame"]+buffer_frame_mean))][p].mean(),axis=1)
            df_m2[p+"_mean"] = df_m2.apply(lambda row:df_mss[(df_mss["id"]==row["id"]) & (df_mss["frame"].between(row["frame"]-buffer_frame_mean,row["frame"]+buffer_frame_mean))][p].mean(),axis=1)

        # m3
        df_m3 = df_mrs[["id","frame_count","TTC"]].merge(df_mss[get_vars],on=["id","TTC"],how="left")
        df_m3 = df_m3.drop_duplicates(subset=["id","TTC"])
        # Get virtual_lane_type
        df_m3 = df_m3.merge(df_mrs_OG,on=["id","frame"],how="left")
        # Mean parameters
        for p in avg_vars:
            df_m3[p+"_mean"] = df_m3.apply(lambda row:df_mss[(df_mss["id"]==row["id"]) & (df_mss["frame"].between(row["frame"]-buffer_frame_mean,row["frame"]+buffer_frame_mean))][p].mean(),axis=1)

        # m4
        m4_par = "median"
        df_mss_filtered_agg = df_mss_filtered.groupby("id").agg({
            "TTC":m4_par,
            "frame":m4_par,
            "vehicle_type_reference":"first",
            "front":m4_par,
            "vehicle_type_front":"first",
            "front_gap":m4_par,
            "x_instant_speed_reference":m4_par,
            "x_instant_speed_front":m4_par,
            "delta_instant_speed_x":m4_par,
            'max_obs_space_headway':m4_par,
            'y_instant_speed_reference':m4_par,
            'x_instant_acc_reference':m4_par,
            'y_instant_speed_front':m4_par,
            'x_instant_acc_front':m4_par,
            'degree_aligment':m4_par,
            'space_headway':m4_par,
            "right":m4_par,
            "dist_right":m4_par,
            "left":m4_par,
            "dist_left":m4_par,
            "real_width_lane":m4_par,
        }).reset_index(drop=False)
        # m4
        df_m4 = df_mrs[["id","frame_count"]].merge(df_mss_filtered_agg[get_vars],on=["id"],how="left")
        df_m4 = df_m4.drop_duplicates(subset=["id"])
        # Get virtual_lane_type
        df_m4 = df_m4.merge(df_mrs_OG,on=["id","frame"],how="left")

        # renames
        df_m1.columns = [col+"_m1" if not col in ["id","frame_count","vehicle_type_reference","virutal_lane","virtual_lane_type"] else col for col in df_m1.columns]
        df_m2.columns = [col+"_m2" if not col in ["id","frame_count","vehicle_type_reference","virutal_lane","virtual_lane_type"] else col for col in df_m2.columns]
        df_m3.columns = [col+"_m3" if not col in ["id","frame_count","vehicle_type_reference","virutal_lane","virtual_lane_type"] else col for col in df_m3.columns]
        df_m4.columns = [col+"_m4" if not col in ["id","frame_count","vehicle_type_reference","virutal_lane","virtual_lane_type"] else col for col in df_m4.columns]
        
        # df_concat = df_m1.merge(df_m2,on=["id","frame_count","vehicle_type_reference"],how="left").merge(df_m3,on=["id"],how="left")
        # df_concat.to_excel("Teste.xlsx",index=False)
        
        df_m1.to_csv(os.path.join("data/collected/Dissertação/Interaction/Motorcycle Lane/m1",f.replace(".json",".csv")),index=False)
        df_m2.to_csv(os.path.join("data/collected/Dissertação/Interaction/Motorcycle Lane/m2",f.replace(".json",".csv")),index=False)
        df_m3.to_csv(os.path.join("data/collected/Dissertação/Interaction/Motorcycle Lane/m3",f.replace(".json",".csv")),index=False)
        df_m4.to_csv(os.path.join("data/collected/Dissertação/Interaction/Motorcycle Lane/m4",f.replace(".json",".csv")),index=False)

        print("OK",f)

    