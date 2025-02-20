import csv 
from collections import defaultdict, Counter, OrderedDict
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import os


folders = []
obj = sorted(os.scandir("Nimba Events"), key = lambda e: e.name)

for item in obj:
    if item.is_dir():
        folders.append(item.name)

# folders.append(name.name for name in os.scandir("Nimba Events") if name.is_dir())
full_list = []

n_rows = 17280
skip = np.arange(n_rows)
# set skip number by hourly samples - 1 (@ 5 sec sampling = 12 samples/min*60min/hour - 1) 
skip = np.delete(skip, np.arange(0, n_rows, 12))

for folder in folders:
    print(folder)
    filePath = f"/Users/aliagola/Documents/capstone tings/code/Nimba Events/{folder}/SD78_Controller.csv"
    file = pd.read_csv(filePath, usecols = ["time_stamp","load_gross_real_power_kW"], skiprows=skip)
    for i in range(0, len(file)):
        if i%60 != 0:
            file.drop([i], inplace=True)
    full_list.append(file)

dataframe = pd.concat(full_list)



# n_rows = 17280*439
# skip2 = np.arange(n_rows)
# # set skip number by hourly samples - 1 (@ 5 sec sampling = 12 samples/min*60min/hour - 1) 
# skip2 = np.delete(skip2, np.arange(0, n_rows, 60))

dataframe.to_csv("load_data.csv", encoding='utf-8', index=False)
