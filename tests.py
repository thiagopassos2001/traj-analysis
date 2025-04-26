from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import json
import os

gdf1 = gpd.GeoDataFrame(["R1","R2"],geometry=[shapely.Polygon([[0,0],[0,10],[10,10],[10,0],[0,0],]),
                                  shapely.Polygon([[20,20],[0,30],[30,30],[30,0],[20,20],])],crs="EPSG:31984")
gdf2 = gpd.GeoDataFrame(["P1","P2"],geometry=[shapely.Point([5,5],),
                                  shapely.Point([5,5],)],crs="EPSG:31984")

print(gdf2.overlay(gdf1,how='union'))
