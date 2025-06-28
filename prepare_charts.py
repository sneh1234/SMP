
import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from PIL import Image, ImageOps
from scipy.stats import gmean



import pickle
import pandas as pd
from PIL import Image, ImageOps




from torchvision import transforms
 
# define custom transform
# here we are using our calculated
# mean & std
transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.97, 0.135)
])

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data_size = {}
"""
for f in os.listdir('Validation_Data/'):
    df = pd.read_csv('Validation_Data/' + f)
    data_size[f] = df[df['Date'] >= '2019-01-01'].shape[0]
"""

#baseline_stock = max(data_size, key=data_size.get)
baseline_stock = 'BAJAJFINSV.NS.csv'

dfx = pd.read_csv('Validation_Data/' + baseline_stock)
required_size = dfx[dfx['Date'] >= '2019-01-01'].shape[0]   
number_of_chunks = dfx[dfx['Date'] >= '2019-01-01'].shape[0] // 50


print(dfx[dfx['Date'] >= '2019-01-01'])
new_paths_400 = []
new_paths = []
c = 0
for f in os.listdir('Validation_Data/'):
    if '.NS.csv' not in f:
        continue
    
    
    df = pd.read_csv('Validation_Data/' + f)
    df['Date'] = df['Date'].apply(lambda x: str(x).split(' ')[0])
 
    turnover = (df.iloc[-1196:]['Close'] * df.iloc[-1196:]['Volume']).sum()  / 1e12
    if turnover >= 0.4 and len(df) >= required_size - 10 :
        print(f.split('.')[0])
    else:
        continue
        
    c = c + 1
    


    print('c:', c, len(df))

    dates = dfx.iloc[-400:][['Date']].copy()
    dates['Date'] = dates['Date'].apply(lambda x: str(x).split(' ')[0])

    rel_date = set(dates['Date'].unique())
    print(df['Date'].unique(), dates)
    df_sub = df[df['Date'].isin(rel_date)].copy()
    print(len(df_sub))



    if len(df_sub) < 390:
        continue
    df_sub = dates.merge(df_sub, on = 'Date', how = 'outer')

    df_sub['Close'] = df_sub['Close'].bfill()
    df_sub['Close'] = df_sub['Close'].ffill()

    st_cl = df_sub['Close'].values[0]

    df_sub['Close'] = df_sub['Close'].apply(lambda x: x / (st_cl))
    df_sub['Close'] = df_sub['Close'].apply(lambda x: np.log(x)/1.5 )


    df_sub = df_sub.iloc[1::2]
    maxi = max(df_sub['Close'])
    mini = min(df_sub['Close'])
    if maxi > 1:
        continue
    if mini < -1:
        continue


    print(maxi, mini)




    plt.plot(df_sub['Close'].values, color = 'black')
    plt.ylim(-1, 1)
    plt.savefig('./validation_plots/' + str(f.split('.')[0])  +'_recommendation'+ '_400png')
    plt.close()

    dates = dfx.iloc[-200:][['Date']].copy()
    dates['Date'] = dates['Date'].apply(lambda x: str(x).split(' ')[0])

    rel_date = set(dates['Date'].unique())

    df_sub = df[df['Date'].isin(rel_date)].copy()

    df_sub = dates.merge(df_sub, on = 'Date', how = 'outer')
    df_sub['Close'] = df_sub['Close'].bfill()
    df_sub['Close'] = df_sub['Close'].ffill()


    st_cl = df_sub['Close'].values[0]

    df_sub['Close'] = df_sub['Close'].apply(lambda x: x / (st_cl))
    df_sub['Close'] = df_sub['Close'].apply(lambda x: np.log(x) )
    maxi = max(df_sub['Close'])
    mini = min(df_sub['Close'])


    if maxi > 1:
        continue
    if mini < -1:
        continue





    plt.plot(df_sub['Close'].values, color = 'black')
    plt.ylim(-1, 1)
    plt.savefig('./validation_plots/' + str(f.split('.')[0])  +'_recommendation'+ '_200png')
    plt.close()


    new_paths_400.append('./validation_plots/' + str(f.split('.')[0])  +'_recommendation'+ '_400png.png')
    new_paths.append('./validation_plots/' + str(f.split('.')[0])  +'_recommendation'+ '_200png.png')

