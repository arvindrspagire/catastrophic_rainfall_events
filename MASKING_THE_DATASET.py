# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 22:41:09 2022

@author: Dell
"""
'''

Creating mask using regionmask module

'''

import xarray as xr
import os
import geopandas as gpd
import regionmask
import numpy as np
import matplotlib.pyplot as plt


#Changing to the working directory
DIR = os.chdir("F:\\district_shape")
#Read the shape file
df =gpd.read_file("F:\\district_shape\\India_Districts2020v2.shp")
## Read the shape from shape file
mah = df[df['S_NAME'] == 'MAHARASHTRA']

#Load the data for coordinates
ds = xr.open_mfdataset("F:\\data\\2000\\3IMERG200006-09.nc")
lon = ds.lon.values
lat = ds.lat.values
mask = regionmask.mask_geopandas(mah,lon,lat)
mask_mah = mask.values
mask_mah[np.isnan(mask_mah)]=0
mask_mah[mask_mah!=0]=1
mask_mah[mask_mah==0]=np.nan
