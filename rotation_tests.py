from model import *
import timeit
import os
import warnings
import shapely
warnings.filterwarnings('ignore')

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

        print(model.DirectionEstimate(59,window_length=15))
        print(model.mpp)
        model.DirectionEstimate(59,window_length=15).to_excel("RotMethod.xlsx",index=False)