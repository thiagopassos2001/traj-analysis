from model import *
import timeit
import os
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

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

root_path = "project/Faixa Azul (Fortaleza)"

start_timer = timeit.default_timer()
if __name__=="__main__":
    os.chdir(root_path)

    df = GetVideoDataset()
    id_coleta_list = ["2025-08-06","2026-05-21"]
    df = df[df["id_coleta"].isin(id_coleta_list)]
    # df = df[df["id_video"].isin(["2025-08-06-01","2025-08-06-02"])]
    print(df)
    # df = df.iloc[:1]

    for index,row in df.iterrows():
        file_name = row["nome_video"]+"_processed.csv"
        raw_file_path=os.path.join(root_path,"raw",row["id_coleta"],file_name)
        file_name=row["id_video"]+"_"+file_name.split(".")[0]
        mpp=row["mpp"]
        image_reference=row["img_ref"]
        id_regions=list(eval(row["id_region"]))
        regions=list(eval(row["regions"]))
        id_conflict_zones=list(eval(row["id_conflict_zones"]))
        conflict_zones=list(eval(row["conflict_zones"]))
        video_heigth=int(row["resolucao"].split(" x ")[1])
        video_width=int(row["resolucao"].split(" x ")[0])
        flip_h=row["flip_h"]
        flip_v=row["flip_v"]
        green_open_time=[int(i) for i in row["green_open_time"].split(",")]
        force_processing=False

        RunDataProcessingType2(
            raw_file_path=raw_file_path,
            file_name=file_name,
            mpp=mpp,
            image_reference=image_reference,
            id_regions=id_regions,
            regions=regions,
            id_conflict_zones=id_conflict_zones,
            conflict_zones=conflict_zones,
            video_heigth=video_heigth,
            video_width=video_width,
            flip_h=flip_h,
            flip_v=flip_v,
            green_open_time=green_open_time,
            force_processing=force_processing)

        stop_timer = timeit.default_timer()
        count_timer = stop_timer - start_timer
        print(f"Execução: {int(count_timer//60):02}:{int(count_timer%60):02} (mm:ss)")