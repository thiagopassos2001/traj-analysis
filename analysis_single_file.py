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
    model.regions.to_file("teste.gpkg")
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
    df_bicycle = df_bicycle.groupby(["id","vehicle_type"]).agg({"frame":["first","last"]}).reset_index(drop=False)
    df_bicycle.columns = ["id","vehicle_type","ff","lf"]

    df_non_bicycle = model.df[(model.df["id"].isin(id_list)) & (model.df["vehicle_type"]!="Bicicleta")]
    df_non_bicycle["geometry"] = df_non_bicycle.apply(lambda row:shapely.Polygon([(row["p1xbb"],row["p1ybb"]),(row["p1xbb"],row["p2ybb"]),(row["p2xbb"],row["p2ybb"]),(row["p2xbb"],row["p1ybb"])]),axis=1)
    df_non_bicycle = gpd.GeoDataFrame(df_non_bicycle,
                                geometry="geometry",
                                crs="EPSG:31984")
    zone_non_bicycle = model.conflict_zones[model.conflict_zones["id"]=="ZA-OM"]
    df_non_bicycle = df_non_bicycle.overlay(zone_non_bicycle.rename(columns={"id":"zone"})[["zone","geometry"]],how='intersection')
    df_non_bicycle = df_non_bicycle.sort_values(by=[model.frame_column,model.id_column])
    df_non_bicycle = df_non_bicycle.drop_duplicates(subset=[model.id_column,model.frame_column],keep="first")
    df_non_bicycle = df_non_bicycle.groupby(["id","vehicle_type"]).agg({"frame":["first","last"]}).reset_index(drop=False)
    df_non_bicycle.columns = ["id","vehicle_type","ff","lf"]

    df_bicycle["id_interaction"] = df_bicycle.apply(lambda row:df_non_bicycle[(df_non_bicycle["ff"]<=row["lf"]) & (df_non_bicycle["lf"]>=row["ff"])]["id"].unique().tolist(),axis=1)
    df_bicycle["interaction"] = df_bicycle["id_interaction"].apply(len).astype(bool)

    df_non_bicycle["id_interaction"] = df_non_bicycle.apply(lambda row:df_bicycle[(df_bicycle["ff"]<=row["lf"]) & (df_bicycle["lf"]>=row["ff"])]["id"].unique().tolist(),axis=1)
    df_non_bicycle["interaction"] = df_non_bicycle["id_interaction"].apply(len).astype(bool)

    # 4 step
    # 1 zone
    df_non_bicycle[["ff_z1","lf_z1"]] = df_non_bicycle[["ff","lf"]]
    df_non_bicycle["delta_time_z1"] = (df_non_bicycle["lf_z1"] - df_non_bicycle["ff_z1"])/model.fps
    df_non_bicycle["length_z1"] = df_non_bicycle.apply(lambda row:shapely.LineString(model.df[(model.df["id"]==row["id"]) & (model.df["frame"].between(row["ff_z1"],row["lf_z1"]))][['x', 'y']].values).length,axis=1)
    df_non_bicycle["average_speed_z1"] = 3.6*(df_non_bicycle["length_z1"]/df_non_bicycle["delta_time_z1"])

    # 2 zone
    df_z2 = model.df.copy()
    df_z2 = df_z2[df_z2["id"].isin(df_non_bicycle["id"].unique().tolist())]
    df_z2["geometry"] = df_z2.apply(lambda row:shapely.Polygon([(row["p1xbb"],row["p1ybb"]),(row["p1xbb"],row["p2ybb"]),(row["p2xbb"],row["p2ybb"]),(row["p2xbb"],row["p1ybb"])]),axis=1)
    df_z2 = gpd.GeoDataFrame(df_z2,
                            geometry="geometry",
                            crs="EPSG:31984")
    zone2 = model.conflict_zones[model.conflict_zones["id"]=="ZC-OM"]
    df_z2 = df_z2.overlay(zone2.rename(columns={"id":"zone"})[["zone","geometry"]],how='intersection')
    df_z2 = df_z2.sort_values(by=[model.frame_column,model.id_column])
    df_z2 = df_z2.drop_duplicates(subset=[model.id_column,model.frame_column],keep="first")
    df_z2 = df_z2.groupby("id").agg({"frame":["first","last"]}).reset_index(drop=False)
    df_z2.columns = ["id","ff_z2","lf_z2"]
    df_non_bicycle = df_non_bicycle.merge(df_z2,on="id",how="left")

    df_non_bicycle["delta_time_z2"] = (df_non_bicycle["lf_z2"] - df_non_bicycle["ff_z2"])/model.fps
    df_non_bicycle["length_z2"] = df_non_bicycle.apply(lambda row:shapely.LineString(model.df[(model.df["id"]==row["id"]) & (model.df["frame"].between(row["ff_z2"],row["lf_z2"]))][['x', 'y']].values).length,axis=1)
    df_non_bicycle["average_speed_z2"] = 3.6*(df_non_bicycle["length_z2"]/df_non_bicycle["delta_time_z2"])

    # 3 zone
    df_z3 = model.df.copy()
    df_z3 = df_z3[df_z3["id"].isin(df_non_bicycle["id"].unique().tolist())]
    df_z3["geometry"] = df_z3.apply(lambda row:shapely.Polygon([(row["p1xbb"],row["p1ybb"]),(row["p1xbb"],row["p2ybb"]),(row["p2xbb"],row["p2ybb"]),(row["p2xbb"],row["p1ybb"])]),axis=1)
    df_z3 = gpd.GeoDataFrame(df_z3,
                            geometry="geometry",
                            crs="EPSG:31984")
    zone3 = model.regions[model.regions["id"]=="Acesso 2"]
    df_z3 = df_z3.overlay(zone3.rename(columns={"id":"zone"})[["zone","geometry"]],how='intersection')
    df_z3 = df_z3.sort_values(by=[model.frame_column,model.id_column])
    df_z3 = df_z3.drop_duplicates(subset=[model.id_column,model.frame_column],keep="first")
    df_z3 = df_z3.groupby("id").agg({"frame":["first","last"]}).reset_index(drop=False)
    df_z3.columns = ["id","ff_z3","lf_z3"]
    df_non_bicycle = df_non_bicycle.merge(df_z3,on="id",how="left")

    df_non_bicycle["delta_time_z3"] = (df_non_bicycle["lf_z3"] - df_non_bicycle["ff_z3"])/model.fps
    df_non_bicycle["length_z3"] = df_non_bicycle.apply(lambda row:shapely.LineString(model.df[(model.df["id"]==row["id"]) & (model.df["frame"].between(row["ff_z3"],row["lf_z3"]))][['x', 'y']].values).length,axis=1)
    df_non_bicycle["average_speed_z3"] = 3.6*(df_non_bicycle["length_z3"]/df_non_bicycle["delta_time_z3"])

    print(df_non_bicycle[["id","average_speed_z1","average_speed_z2","average_speed_z3"]].sort_values("average_speed_z1"))

stop_timer = timeit.default_timer()
count_timer = stop_timer - start_timer
print(f"Execução: {int(count_timer//60):02}:{int(count_timer%60):02} (mm:ss)")