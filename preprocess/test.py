import skimage.io as io
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

CASIA_Iris_Africa = pickle.load(open('CASIA_Iris_Africa_split_20210305.pkl', 'rb'))
CASIA_Iris_Asia = pickle.load(open('CASIA_Iris_Asia_split_20210305.pkl', 'rb'))
CASIA_Iris_M1 = pickle.load(open('CASIA_Iris_M1_split_20210305.pkl', 'rb'))

CASIA_Iris_All = []

for i in range(5):
    CASIA_Iris_All.append({'train': CASIA_Iris_Africa[i]['train'] + CASIA_Iris_Asia[i]['train'] +
                            CASIA_Iris_M1[i]['train'], 'val': CASIA_Iris_Africa[i]['val'] +
                            CASIA_Iris_Asia[i]['val'] + CASIA_Iris_M1[i]['val'], 'test': CASIA_Iris_Africa[i]['test'] +
                            CASIA_Iris_Asia[i]['test'] + CASIA_Iris_M1[i]['test']})

print(CASIA_Iris_All)
with open(r'CASIA_Iris_All_split_20210305.pkl', 'wb') as f:
    pickle.dump(CASIA_Iris_All, f, pickle.HIGHEST_PROTOCOL)