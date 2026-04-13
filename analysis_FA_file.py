from model import *
import timeit
import os
import warnings
import shapely
from scipy import stats
warnings.filterwarnings('ignore')

root_path = "project/Safe Lane"
file_name = "DJI_0001_transformed_processed.json"
df_concat = []
start_timer = timeit.default_timer()

if __name__=="__main__":
    root_path = "data_ignore"
    os.chdir(root_path)

    # Início do loop
    for f in os.listdir("data/json"):
        model = YoloMicroscopicDataProcessing()
        model.ImportFromJSON(f"data/json/{f}",post_processing=model.PostProcessing1)

        vl_df = pd.read_csv(f"data/collected/motorcycle_virtual_lane_backup/{f.replace(".json",".csv")}")
        vl_df = vl_df[vl_df["virtual_lane_type"]=="Corredor Principal"]

        fr2side_df = pd.read_csv(f"data/collected/motorcycle_fr2side/{f.replace(".json",".csv")}")
        fr2side_df[["id","frame"]] = fr2side_df[["id","frame"]].astype(int)

        vl_df = vl_df.merge(fr2side_df,on=["id","frame"],how="left")
        # vl_df = vl_df.merge(model.df,on=["id","frame"],how="left")
        
        df_agg = vl_df.groupby("id").agg({
            "frame":["count",stats.mode],
            "dist_front":"min"
        }).reset_index(drop=False).fillna(-1)
        df_agg.columns = ["id","count_frame","frame","dist_front"]
        df_agg["frame"] = df_agg["frame"].apply(lambda x:x[0])
        df_agg = df_agg.merge(model.df[["id","frame","instant_speed"]],on=["id","frame"],how="left")
        df_agg = df_agg.rename(columns={"frame":"mid_frame","instant_speed":"mid_instant_speed"})

        df_agg = df_agg.merge(vl_df[["id","dist_front","frame"]],on=["id","dist_front"],how="left").fillna(-1)
        df_agg["frame"] = df_agg["frame"].astype(int)
        df_agg = df_agg.merge(model.df[["id","frame","instant_speed"]],on=["id","frame"],how="left")
        df_agg = df_agg.rename(columns={"frame":"min_dist_frame"})

        df_agg.to_csv(f"data/collected/V2/{f.replace(".json",".csv")}",index=False)
        print(f.replace(".json",".csv"))