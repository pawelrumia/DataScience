import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import bar_chart_race as bcr
import os
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

drivers = pd.read_csv("C:\\Users\\mazurp2\\PycharmProjects\\DataScience\\resources\\f1\\drivers.csv")



print(drivers.head(5))
print(drivers.columns)