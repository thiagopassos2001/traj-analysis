from model import *
import timeit
import os
import warnings
import shapely
from scipy import stats
warnings.filterwarnings('ignore')

root_path = "project/Faixa Azul SP"
file_name = "10_A_1.json"
df_concat = []
start_timer = timeit.default_timer()

if __name__=="__main__":
    os.chdir(root_path)

    model = YoloMicroscopicDataProcessing()
    model.ImportFromJSON(f"data/json/{file_name}",post_processing=model.PostProcessing1)

    id_ = 445

    # model.AvgVehicleSize()
    # print(model.avg_vehicle_size[model.avg_vehicle_size["id"]==id_])    

    model.SafetySpaceEllipse(
        x=50,
        y=50,
        Wa=0.5,
        dy=0.8,
        dx=2.4,
        taua=0.5,
        va=80/3.6
    )