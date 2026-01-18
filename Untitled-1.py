import numpy as np
import pandas as pd
from model import *
import timeit
import os
import warnings
import shapely
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

model = YoloMicroscopicDataProcessing()
model.ImportFromJSON(os.path.join("data/json","C_x_13M_D2_0004.json"),post_processing=model.PostProcessing1)
print(model.df[model.df["id"]==263]["x_instant_speed"].abs().max(),len(model.df[model.df["id"]==263]))

model.df = model.SmoothingSavGolFilter(window_length=15,polyorder=1)

print(model.df[model.df["id"]==263]["x_instant_speed"].abs().max(),len(model.df[model.df["id"]==263]))

# print(model.df[model.df["id"]==263].head(50)[["frame","x_instant_speed"]])