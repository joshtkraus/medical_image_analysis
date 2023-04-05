#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# NIH Data

# Libraries
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
from npy_append_array import NpyAppendArray
import os

# Data Entry .csv file
## Read
data = pd.read_csv('data/Data_Entry_2017.csv')
## image paths
paths = {os.path.basename(x): x for x in 
         glob(os.path.join('data', 'images*', '*', '*.png'))}
data['path'] = data['Image Index'].map(paths.get)
## length
scan_len = len(data)
print('Total Scans:', scan_len)  

# func to create array format
def create_data(res):
    print('Creating {}x{} Images...'.format(res,res))
    # create folder
    if not os.path.exists('array_data'):
        os.mkdir('array_data')
    # create file
    with NpyAppendArray('array_data/data_{}.npy'.format(res)) as f:
        # iterate through images
        for path in data['path'][:10]:
            # read file
            img = Image.open(path)
            # resize
            img = img.resize((res,res))
            # to array
            img = np.array(img)
            # remove 3rd dimension, if it exists
            if len(img.shape) == 3:
                img = img[:,:,0]
            # write        
            f.append(img.flatten() / 255.)

    # Image Data
    ## open file
    data_file = np.load('array_data/data_{}.npy'.format(res),allow_pickle=True)
    ## reconstruct
    img_data = data_file.reshape((10, res*res))
    ## write
    np.save('array_data/data_{}.npy'.format(res),img_data.T) 

    # Response Array
    ## diseases 
    resp_arr = np.array(data['Finding Labels'])
    ## write
    np.save('array_data/labels.npy',resp_arr)

# 64 x 64 Res
create_data(res=64)

# 128 x 128 Res
create_data(res=138)

