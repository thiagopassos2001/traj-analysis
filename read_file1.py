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
    model.AvgVehicleSize()

    df_space_speed = pd.read_csv(f"data/collected/Dissertação/MotorcycleSpaceSpeed2/{file_name.replace('.json','.csv')}")
    df_space_speed = df_space_speed.merge(model.avg_vehicle_size,on="id",how="left").rename(columns={
        "vehicle_length":"vl_reference",
        "vehicle_width":"vw_reference"}
        )
    df_space_speed = df_space_speed.merge(model.avg_vehicle_size.rename(columns={"id":"left"}),on="left",how="left").rename(columns={
            "vehicle_length":"vl_left",
            "vehicle_width":"vw_left"}
            )
    df_space_speed = df_space_speed.merge(model.avg_vehicle_size.rename(columns={"id":"right"}),on="right",how="left").rename(columns={
            "vehicle_length":"vl_right",
            "vehicle_width":"vw_right"}
            )
    df_space_speed["Wa_right"] = df_space_speed["GapW_right"] - df_space_speed["vw_reference"]*0.5 - df_space_speed["vw_right"]*0.5
    df_space_speed["Wa_left"] = df_space_speed["GapW_left"] - df_space_speed["vw_reference"]*0.5 - df_space_speed["vw_left"]*0.5
    df_space_speed[["Wa_right","Wa_left"]] = df_space_speed[["Wa_right","Wa_left"]].fillna(999)
    
    df_space_speed["Wa_min"] = df_space_speed.apply(lambda row:min(row["Wa_right"],row["Wa_left"]),axis=1)

    df_space_speed = df_space_speed[(-df_space_speed["GapW_left"].isna()) | (-df_space_speed["GapW_right"].isna())]
    # df_space_speed = df_space_speed[df_space_speed["Wa_min"]<=2]
    # print(df_space_speed[["vw_reference","GapW_right","vw_right","Wa_right","GapW_left","vw_left","Wa_left","Wa_min"]])

    df_space_speed = df_space_speed.groupby("id").agg({"Wa_min":"min"}).reset_index()

    print(df_space_speed["Wa_min"].describe())
    df_space_speed["Wa_min"].hist()
    plt.show()
    
