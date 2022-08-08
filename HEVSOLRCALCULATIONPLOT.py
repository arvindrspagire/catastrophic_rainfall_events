# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 11:34:05 2022

@author: arvindp
"""
########### Load the modules #################### 
import xarray as xr
import os
import glob
import h5py
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm
#import h5py
import numpy as np
import geopandas as gpd
from pyresample import geometry
import cmaps
from pyresample.kd_tree import resample_nearest
from matplotlib import colors 
import pandas as pd
import seaborn as sns
import cartopy.feature as cfeature
#import h5py

################ Function to truncate Colormap #####
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

############### Read the File ########################### 
fn =h5py.File("D:\\OLR\\Jun22_075381\\3DIMG_02JUN2022_0000_L3B_OLR_DLY.h5")
################ Read the Shape File ####################
df = gpd.read_file('E:\\Data\\06-09_2000-2021\\state_shape\\India_State_Boundary.shp')
#################################### Read the Variables #####################
print(list(fn.keys()))
print('Start on RF Plotting for time series')
###Lat Lon data for location of the cloudburst events
#Amarnath Date |(8/July:4.30-6.00 PM) | Lat: 34.21 & | Lon: 75.50
#Doda  Date (8/July:2.00AM-3.00 AM )   | Lat: 33.30 & | Lon: 75.10
#Parbati Valley (6/July early hours)|Lat:32.13 & |Lon: 77.10 
#Sherinala,(25/July) Himachal Pradesh 31.29, 77.69
#Tanta Area of Kahara Tehsil,(20/July) Doda ()33.07, 75.85
#Shalkhar village of Kinnaur (18/July) 32.00, 78.57
#Sobla, Dharucha Pithorgarh (31st July) 29.84, 80.53



def lat_index(lati,loni):
    yi =lati-20
    yf =lati+20
    xi =loni-20
    xf =loni+20
    return yi,yf,xi,xf

#Extraction of RF data
path_hem = "D:\\INSAT\\HEM\\"
rf_20 =np.empty([48,2816,2805])
rf_shalkhar =np.empty([48,40,40])
rf_sherinala=np.empty([48,40,40])
rf_tanta = np.empty([48,40,40])

#Change the date accordingly
for i, filename in enumerate(sorted(glob.glob(os.path.join(path_hem,'3DIMG_20*.h5')))):
    print(i)
    print(filename)
    ds = h5py.File(filename)
    rf= np.asarray(ds['HEM'])
    time=np.asarray(ds['time'])
    rf[rf<0]=np.nan
    if i==0:
        # print(i)
        # print(filename)
        # ds = h5py.File(filename)
        # rf= np.asarray(ds['HEM'])

        # time=np.asarray(ds['time'])
        # print(time)
        # rf[rf<0]=np.nan
        lat =np.asarray(ds['Latitude'])
        lon =np.asarray(ds['Longitude'])
        lat = np.where(lat==32767, np.nan, lat)*0.01
        lon = np.where(lon==32767, np.nan,lon)*0.01
  
        oldLonLat = geometry.SwathDefinition(lons =lon, lats =lat)
        x =np.linspace(np.nanmin(lon), np.nanmax(lon),2805)
        y =np.linspace(np.nanmin(lat), np.nanmax(lat),2816)
        #Following lines can be used while extracting the dataset for
        #finding the index of location
  
        #4 Sherinala 25/July
        lati_s=np.abs(y-31.29).argmin()
        loni_s=np.abs(x-77.69).argmin()
        #5 Tanta Area 20 July
        lati_t=np.abs(y-33.07).argmin()
        loni_t=np.abs(x-75.85).argmin()
        #6 Shalkhar village 18/July
        lati_k=np.abs(y-32.00).argmin()
        loni_k=np.abs(x-78.57).argmin()
        
        
        lati=np.abs(y-33.07).argmin()
        loni=np.abs(x-75.5041).argmin()
   
        ind_t =lat_index(lati_t,loni_t)
        ind_s =lat_index(lati_s, loni_s)
        ind_sh=lat_index(lati_k,loni_k)
        
        newLon, newLat = np.meshgrid(x,y)
        newLonLat =geometry.GridDefinition(lons =newLon, lats=newLat)
        
    rf =resample_nearest(oldLonLat, rf, newLonLat, radius_of_influence=5000, fill_value=np.nan)
    rf_tanta[i,:,:] =rf[ind_t[0]:ind_t[1],ind_t[2]:ind_t[3]]
  #  rf_sherinala[i,:,:]=rf[ind_s[0]:ind_s[1],ind_s[2]:ind_s[3]]
   # rf_shalkhar[i,:,:]=rf[ind_sh[0]:ind_sh[1], ind_sh[2]:ind_sh[3]]


rf_tanta =rf_tanta[:,38,1]



cld_id = np.where(rf_tanta==np.nanmax(rf_tanta))
cld_id_sh =np.where(rf_sherinala==np.nanmax(rf_sherinala))

print(cld_id)

rf_cld_tanta = rf_tanta[:,10:30,10:30]
rf_serinala =rf_sherinala[:,38,36]

cld_id = np.where(rf_cld_tanta==np.nanmax(rf_cld_tanta))


import numpy as np
###Load the rf extracted array
sh = np.load("D:\\OLR\\Cloudburst\\olr\\rf_serinala.npy")
kn = np.load("D:\\OLR\\Cloudburst\\olr\\rf_shalkhar_cld_3d.npy")
tt =np.load("D:\\OLR\\Cloudburst\\olr\\rf_tanta_final.npy")
# Load the olr extracted array
sh1  = np.load("D:\\OLR\\Cloudburst\\olr\\olr_sherianal.npy")
tt1 = np.load("D:\\OLR\\Cloudburst\\olr\\olr_sherianal.npy")
kn1  = np.load("D:\\OLR\\Cloudburst\\olr\\olr_shalkar.npy")

tantarf = tt[:,16,13]


# OLR Extraction
path = "D:\\INSAT\\OLR\\"
olr_shalkar = np.empty([48,40,40])
olr_sherinala=np.empty([48,40,40])
olr_tanta =np.empty([48,40,40])

# Change the date accordingly




for i, filename in enumerate(sorted(glob.glob(os.path.join(path, '3DIMG_20*.h5')))):
    print(i)
    print(filename)
    ds = h5py.File(filename)
    olr= np.asarray(ds['OLR'])
    time=np.asarray(ds['time'])
    print(time)
    olr[olr<0]=np.nan
    if i==0:
        lat =np.asarray(ds['Latitude'])
        lon =np.asarray(ds['Longitude'])
        lat = np.where(lat==32767, np.nan, lat)*0.01
        lon = np.where(lon==32767, np.nan,lon)*0.01
  
        oldLonLat = geometry.SwathDefinition(lons =lon, lats =lat)
        x =np.linspace(np.nanmin(lon), np.nanmax(lon),2805)
        y =np.linspace(np.nanmin(lat), np.nanmax(lat),2816)
        lati=np.abs(y-32.0185).argmin()
        loni=np.abs(x-77.3279).argmin()

        #Following lines can be used while extracting the dataset for
        #finding the index of location
  
        #4 Sherinala 25/July
        lati_s=np.abs(y-31.29).argmin()
        loni_s=np.abs(x-77.69).argmin()
        #5 Tanta Area 20 July
        lati_t=np.abs(y-33.07).argmin()
        loni_t=np.abs(x-75.85).argmin()
        #6 Shalkhar village 18/July
        lati_k=np.abs(y-32.00).argmin()
        loni_k=np.abs(x-78.57).argmin()
        
        
        lati=np.abs(y-33.07).argmin()
        loni=np.abs(x-75.5041).argmin()
        ind_s =lat_index(lati_s,loni_s)
        ind_t =lat_index(lati_t,loni_t)
        ind_sh=lat_index(lati_k,loni_k)
        
        
        newLon, newLat = np.meshgrid(x,y)
        newLonLat =geometry.GridDefinition(lons =newLon, lats=newLat)
        
    OLR =resample_nearest(oldLonLat, olr, newLonLat, radius_of_influence=10000, fill_value=np.nan)
 #   olr_shalkar[i,:,:]=OLR[ind_s[0]:ind_s[1], ind_s[2]:ind_s[3]]
#    olr_sherinala[i,:,:]=OLR[ind_sh[0]:ind_sh[1], ind_sh[2]:ind_sh[3]]
    olr_tanta[i,:,:]= OLR[ind_t[0]:ind_t[1],ind_t[2]:ind_t[3]]








olr_cld_sh =olr_sherinala[:,20,20]

olr_tt = olr_tanta[:,38,1]



    







index =lat_index(lati,loni)














lati=np.abs(y-32.00).argmin()
loni=np.abs(x-78.57).argmin()
yi =lati-20
yf =lati+20
xi =loni-20
xf =loni+20
rf_tanta =rf_20[:,yi:yf, xi:xf]
rf_sher =rf_20[:,yi:yf,xi:xf]
rf_shalkhar =rf_20[:,yi:yf, xi:xf]

lon_a =newLon[0,xi:xf]
lat_a =newLat[yi:yf,0]

#rf_hem = np.nansum(rf_6, axis=0)
#rf_hem[rf_hem==0.0]= np.nan



rf_sher[rf_sher==0]= np.nan
rf_shalkhar[rf_shalkhar==0]=np.nan
rf_shal_cld = np.nan_to_num(rf_shalkhar[:,23,21])
for i in range(46):
    plt.pcolormesh(lon_a, lat_a, rf_shalkhar[i,:,:], cmap ='PRGn', vmin=0.1, vmax=100)
    plt.colorbar(shrink=0.6)
    plt.plot(78.57,32.00,'*', color='red')
    plt.title(''+str(i)+'')
    plt.show()






        


 
    


        
            
        



        




time_ist =pd.date_range('2022-07-08-05-30-00', periods=46, freq='0.5H')

for i in range(46):
    if i==0:
        
        levels = [0.1,5,10,15,20,25,30,35,40,45,50,55,60,70,75,80,85,90,95,100,105,110]
        cmap = plt.get_cmap('jet')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
       
        scale ='110m'
        fig = plt.figure(figsize=(5,5))
        sns.set_theme(style="white", palette=None)
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines(scale)
        ax.set_extent([72,78,31.5,37])
        plt.pcolormesh(lon,lat,rf , transform=ccrs.PlateCarree(),cmap =truncate_colormap(cmaps.WhBlGrYeRe, 0.4, 0.9), norm= norm)
        
        df.plot(ax=ax, edgecolor='black',linewidth = 2.0, facecolor ='None')
        gls =ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
          linewidth=1.0, color='gray', linestyle='--')
        plt.colorbar(shrink=0.6)
      #  ax.set_title('Accumulated rainfall (mm) derived from INSAT on\n'+str(time_insat[i])+'UTC', fontname="Times New Roman", weight="bold")
        
        gls.xlabels_top = False
        gls.ylabels_right = False
        gls.xlabel_style = {'size': 9, 'weight':'bold'}
        gls.ylabel_style = {'size':9, 'weight':'bold'}
        fig1 = plt.gcf()
         #fig1.savefig('ERE_freq_99.pdf', dpi =300)
        fig1.savefig('D:\\OLR\\Cloudburst\\olr\\olr_anom_himachal_'+str(i)+'.png', dpi=600)
      #  images.append(imageio.imread('D:\\OLR\\Cloudburst\\olr\\olr_anom_himachal_'+str(i)+'.png'))
        
        plt.show()

import numpy.ma as ma

##### Cloudburst with elevation ##########
ds1 =xr.open_dataset("E:\\Data\\elevation\\elevation_nwh_1.nc")
ht =  ds1.ele.values
lat1= ds1.lat.values
lon1 = ds1.lon.values
##########################################
###Lat Lon data for location of the cloudburst events
#Amarnath Date |(8/July:4.30-6.00 PM) | Lat: 34.21 & | Lon: 75.50
#Doda  Date (8/July:2.00AM-3.00 AM )   | Lat: 33.30 & | Lon: 75.10
#Parbati Valley (6/July early hours)|Lat:32.13 & |Lon: 77.10 
#Sherinala,(25/July) Himachal Pradesh 31.29, 77.69
#Tanta Area of Kahara Tehsil,(20/July) Doda ()33.07, 75.85
#Shalkhar village of Kinnaur (18/July) 32.00, 78.57

########################################################
# Function for finding the index
df = gpd.read_file('E:\\Data\\06-09_2000-2021\\state_shape\\India_State_Boundary.shp')

def index(lon1,lat1):
    lati =np.abs(lat1-34.21).argmin()
    loni =np.abs(lon1-75.50).argmin()
    return lati, loni
##
x,y=34.21,75.50


amar = np.abs(lon1-75.50).argmin(), np.abs(lat1-34.21).argmin()
doda = np.abs(lon1-75.10).argmin(), np.abs(lat1-33.30).argmin()
parbati = np.abs(lon1-77.10).argmin(), np.abs(lat1-32.13).argmin()
sher = np.abs(lon1-77.69).argmin(), np.abs(lat1-31.29).argmin()
tanta = np.abs(lon1-75.85).argmin(), np.abs(lat1-33.07).argmin()
shal = np.abs(lon1-78.57).argmin(), np.abs(lat1-32.00).argmin()
#amar = np.abs(lon1-75.50).argmin(), np.abs(lat1-34.21).argmin()


ele_am =ht[0,amar[1], amar[0]]
ele_doda=ht[0,doda[1],doda[0]]
ele_parbati=ht[0,parbati[1],parbati[0]]
ele_sher=ht[0,sher[1],sher[0]]
ele_tanta=ht[0,tanta[1],tanta[0]]
ele_shal=ht[0,shal[1],shal[0]]

elevation =np.asarray([ele_am, ele_doda, ele_parbati, ele_sher,ele_tanta,ele_shal])



scale ='110m'
fig = plt.figure(figsize=(5,5))
sns.set_theme(style="white", palette=None)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(scale)
ax.set_extent([72,81,28,38])
#plt.pcolormesh(lon1,lat1,ht[0,:,:], transform=ccrs.PlateCarree(),cmap =truncate_colormap(cmaps.cmp_b2r, 0.1, 0.9))
plt.plot(75.50,34.21, '*', color='red', label='Amarnath')
plt.plot(75.10,33.30,'*', color='red', label='Doda')
plt.plot(77.10,32.13,'*',color='red',label='Parbati Valley')
plt.plot(77.69,31.29,'*',color='red', label='sherinala')
plt.plot(75.85,33.07,'*', color='red', label='Kahara')
plt.plot(78.57,32.00,'*',color='red',label='Shalkhar')
#plt.legend()






df.loc[(df['Name'] == 'Uttarakhand') | (df['Name'] == 'Jammu and Kashmir') | (df['Name'] =='Himachal Pradesh') | (df['Name'] =='Ladakh')].plot(ax=ax, edgecolor='black',linewidth = 1.5, facecolor ='None')

gls =ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
  linewidth=1.0, color='gray', linestyle='--')
#plt.colorbar(shrink=0.6)
#ax.set_title('OLR Difference on\n'+str(time_insat[i])+'UTC', fontname="Times New Roman", weight="bold")

gls.xlabels_top = False
gls.ylabels_right = False
gls.xlabel_style = {'size': 9, 'weight':'bold'}
gls.ylabel_style = {'size':9, 'weight':'bold'}
plt.show()



        












    










