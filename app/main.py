import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "NASA_Asteroid_Close_Approaches.csv")

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

print("----LOADING DATA----")

dataframe_imported = pd.read_csv(DATA_FILE)
print(dataframe_imported.head(10))

dataframe = dataframe_imported.copy()
dataframe["cd_parsed"] = pd.to_datetime(dataframe["cd"], format="%Y-%b-%d %H:%M", errors="coerce")
dataframe["year"]   = dataframe["cd_parsed"].dt.year
dataframe["month"]  = dataframe["cd_parsed"].dt.month
dataframe["is_pha"] = dataframe["h"] < 22          # flag de Potentially Hazardous Asteroid
dataframe["dist_ld"] = dataframe["dist"] * 389.17  # distância em Distâncias Lunares