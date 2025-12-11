 # Libs principais
from model import *
from TTC_model import *
import pandas as pd
from scipy.signal import savgol_filter
import timeit

# Controle de execução e pastas
import os

# Desativar alguns warnings
import warnings
warnings.filterwarnings('ignore')



start_timer = timeit.default_timer()