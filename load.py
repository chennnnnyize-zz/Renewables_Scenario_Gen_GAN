import sys
from numpy import shape
import csv
sys.path.append('..')
import numpy as np
import os

#Load .csv renewables data into GANs model
#Currently use power generation historical data from NREL, which can be downloaded from NREL wind or solar integration datasets
#Historical data are loaded by column sequence and reshape into model input shape, which is adjustable
#Label is only useful for event-based scenario generation

def load_wind():
    #data created on July 8th, WA 66 wind farms
    with open('new.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=float)
    trX = []
    print(shape(rows))
    m = np.ndarray.max(rows)
    print("Maximum value of wind", m)
    print(shape(rows))
    for x in range(rows.shape[1]):
        train = rows[:, x].reshape(-1, 576)
        train = train / 16

        # print(shape(train))
        if trX == []:
            trX = train
        else:
            trX = np.concatenate((trX, train), axis=0)
    print("Shape TrX", shape(trX))

    with open('sample label.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    label = np.array(rows, dtype=int)
    label=label.T
    print("Label shape", shape(label))
    return trX, label




