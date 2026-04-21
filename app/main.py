import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "NASA_Asteroid_Close_Approaches.csv")

print("----LOADING DATA----")

dataframe = pd.read_csv(DATA_FILE)
print(dataframe.head(10))