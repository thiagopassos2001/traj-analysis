from model import *
import timeit
import os
import warnings
import shapely
warnings.filterwarnings('ignore')

root_path = "project/Safe Lane"
file_name = "DJI_0001_transformed_processed.json"

start_timer = timeit.default_timer()
if __name__=="__main__":
    os.chdir(root_path)

    model = YoloMicroscopicDataProcessing()
    model.ImportFromJSON2(os.path.join("data/json",file_name))

    # Gambs
    drop_id = model.df[model.df["vehicle_type"]=="Pedestre"]["id"].unique().tolist()
    drop_id = drop_id + model.df[model.df["traffic_region"]=="Acesso 1"]["id"].unique().tolist()
    model.df = model.df[-(model.df["id"].isin(drop_id))]

    # 1 step
    zone = model.conflict_zones[model.conflict_zones["id"]=="ZC1"]
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
    zone = model.conflict_zones[-(model.conflict_zones["id"]=="ZC1")]
    df_filtered = gpd.GeoDataFrame(df_filtered,
                                geometry=gpd.points_from_xy(df_filtered[model.x_centroid_column],df_filtered[model.y_centroid_column]),
                                crs="EPSG:31984")
    df_filtered = df_filtered.overlay(zone.rename(columns={"id":"zone"})[["zone","geometry"]],how='intersection')
    df_filtered = df_filtered.sort_values(by=[model.frame_column,model.id_column])
    df_filtered = df_filtered.drop_duplicates(subset=[model.id_column,model.frame_column],keep="first").drop_duplicates(subset=[model.id_column],keep="first")
    id_list = df_filtered["id"].tolist()
    print(f"Filtro 2: {len(id_list)}")

    # 3 step
    model.SaveImg(id_list)
    df_bicycle = model.df[(model.df["id"].isin(id_list)) & (model.df["vehicle_type"]=="Bicicleta")]
    df_bicycle["geometry"] = df_bicycle.apply(lambda row:shapely.Polygon([(row["p1xbb"],row["p1ybb"]),(row["p1xbb"],row["p2ybb"]),(row["p2xbb"],row["p2ybb"]),(row["p2xbb"],row["p1ybb"])]),axis=1)
    df_bicycle = gpd.GeoDataFrame(df_bicycle,
                                geometry="geometry",
                                crs="EPSG:31984")
    zone_bicycle = model.conflict_zones[model.conflict_zones["id"]=="ZA-B"]
    df_bicycle = df_bicycle.overlay(zone_bicycle.rename(columns={"id":"zone"})[["zone","geometry"]],how='intersection')
    df_bicycle = df_bicycle.sort_values(by=[model.frame_column,model.id_column])
    df_bicycle = df_bicycle.drop_duplicates(subset=[model.id_column,model.frame_column],keep="first")
    df_bicycle = df_bicycle.groupby("id").agg({"frame":["first","last"]}).reset_index(drop=False)
    df_bicycle.columns = ["id","ff","fl"]

    df_non_bicycle = model.df[(model.df["id"].isin(id_list)) & (model.df["vehicle_type"]!="Bicicleta")]
    df_non_bicycle["geometry"] = df_non_bicycle.apply(lambda row:shapely.Polygon([(row["p1xbb"],row["p1ybb"]),(row["p1xbb"],row["p2ybb"]),(row["p2xbb"],row["p2ybb"]),(row["p2xbb"],row["p1ybb"])]),axis=1)
    df_non_bicycle = gpd.GeoDataFrame(df_non_bicycle,
                                geometry="geometry",
                                crs="EPSG:31984")
    zone_non_bicycle = model.conflict_zones[model.conflict_zones["id"]=="ZA-OM"]
    df_non_bicycle = df_non_bicycle.overlay(zone_non_bicycle.rename(columns={"id":"zone"})[["zone","geometry"]],how='intersection')
    df_non_bicycle = df_non_bicycle.sort_values(by=[model.frame_column,model.id_column])
    df_non_bicycle = df_non_bicycle.drop_duplicates(subset=[model.id_column,model.frame_column],keep="first")
    df_non_bicycle = df_non_bicycle.groupby("id").agg({"frame":["first","last"]}).reset_index(drop=False)
    df_non_bicycle.columns = ["id","ff","fl"]

    df_bicycle["id_interaction"] = df_bicycle.apply(lambda row:df_non_bicycle[(df_non_bicycle["ff"]<=row["fl"]) & (df_non_bicycle["fl"]>=row["ff"])]["id"].unique().tolist(),axis=1)
    df_bicycle["interaction"] = df_bicycle["id_interaction"].apply(len).astype(bool)

    df_non_bicycle["id_interaction"] = df_non_bicycle.apply(lambda row:df_bicycle[(df_bicycle["ff"]<=row["fl"]) & (df_bicycle["fl"]>=row["ff"])]["id"].unique().tolist(),axis=1)
    df_non_bicycle["interaction"] = df_non_bicycle["id_interaction"].apply(len).astype(bool)

stop_timer = timeit.default_timer()
count_timer = stop_timer - start_timer
print(f"Execução: {int(count_timer//60):02}:{int(count_timer%60):02} (mm:ss)")