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
        "DJI_0002_transformed_processed.json",
        "DJI_0003_transformed_processed.json",
        "DJI_0004_transformed_processed.json",
        "DJI_0005_transformed_processed.json",
        "DJI_0008_transformed_processed.json"
    ]:
        model = YoloMicroscopicDataProcessing()
        model.ImportFromJSON2(os.path.join("data/json",file_name))

        # Gambs
        drop_id = model.df[model.df["vehicle_type"]=="Pedestre"]["id"].unique().tolist()
        drop_id = drop_id + model.df[model.df["traffic_region"]=="Acesso 1"]["id"].unique().tolist()
        model.df = model.df[-(model.df["id"].isin(drop_id))]

        # 1 step
        zone = model.conflict_zones[model.conflict_zones["id"]=="ZConf1"]
        df_filtered = gpd.GeoDataFrame(model.df,
                                    geometry=gpd.points_from_xy(model.df[model.x_centroid_column],model.df[model.y_centroid_column]),
                                    crs="EPSG:31984")
        df_filtered = df_filtered.overlay(zone.rename(columns={"id":"zone"})[["zone","geometry"]],how='intersection')
        df_filtered = df_filtered.sort_values(by=[model.frame_column,model.id_column])
        df_filtered = df_filtered.drop_duplicates(subset=[model.id_column,model.frame_column],keep="first").drop_duplicates(subset=[model.id_column],keep="first")
        id_list = df_filtered["id"].tolist()
        print(f"Filtro 1: {len(id_list)}")

        # 2 step
        df_filtered = model.df.copy()
        df_filtered = df_filtered[df_filtered["id"].isin(id_list)]
        zone = model.conflict_zones[-(model.conflict_zones["id"]=="ZConf1")]
        df_filtered = gpd.GeoDataFrame(df_filtered,
                                    geometry=gpd.points_from_xy(df_filtered[model.x_centroid_column],df_filtered[model.y_centroid_column]),
                                    crs="EPSG:31984")
        df_filtered = df_filtered.overlay(zone.rename(columns={"id":"zone"})[["zone","geometry"]],how='intersection')
        df_filtered = df_filtered.sort_values(by=[model.frame_column,model.id_column])
        df_filtered = df_filtered.drop_duplicates(subset=[model.id_column,model.frame_column],keep="first").drop_duplicates(subset=[model.id_column],keep="first")
        id_list = df_filtered["id"].tolist()
        print(f"Filtro 2: {len(id_list)}")

        df_ = model.df[model.df["id"].isin(id_list)]
        df_["id"] = df_["id"].astype(str) + "_" + file_name
        
        df_concat.append(df_)
        print(file_name)

    # 3 step
model.df = pd.concat(df_concat,ignore_index=True)
model.SaveImg()

stop_timer = timeit.default_timer()
count_timer = stop_timer - start_timer
print(f"Execução: {int(count_timer//60):02}:{int(count_timer%60):02} (mm:ss)")