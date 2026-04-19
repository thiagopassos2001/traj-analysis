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
        output_file_path = os.path.join("data/collected/Dissertação/SpeedFlowDensity",f.replace(".json",".csv"))
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

            result = []
            step = 300
            for lane in traffic_lane_list:
                result_ = model.SpeedFlowDensityAgg(step=step,traffic_lane_list=[lane])
                result_.insert(0,"traffic_lane",lane)
                result.append(result_)

            result = pd.concat(result,ignore_index=True)

            max_step = result["step"].max()
            num_sec = model.df[model.frame_column].max()/model.fps
            max_step_value = num_sec if step > num_sec else num_sec % step
            result["delta_time"] = result.apply(lambda row: step/3600 if row["step"]<max_step else max_step_value/3600,axis=1)
            result["flow"] = result["count"] / result["delta_time"]
            result["density"] = result["flow"] / result["speed"]

            result.to_csv(output_file_path,index=False)

            print("OK",f)
        else:
            print("Já processado",f)
    

