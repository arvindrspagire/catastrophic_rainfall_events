#!/usr/bin/env python
# coding: utf-8



from rasterio import features
from affine import Affine

import geopandas as gpd
import xarray as xr

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors 
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm

#################(2)##################################
def transform_from_latlon(lat, lon):
    """ input 1D array of lat / lon and output an Affine transformation
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale

def rasterize(shapes, coords, latitude='lat', longitude='lon',
              fill=np.nan, **kwargs):
  
  
    transform = transform_from_latlon(coords[latitude], coords[longitude])
    out_shape = (len(coords[latitude]), len(coords[longitude]))
    raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
    spatial_coords = {latitude: coords[latitude], longitude: coords[longitude]}
    return xr.DataArray(raster, coords=spatial_coords, dims=(latitude, longitude))

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
########### shape file read and extract states #####################


states = gpd.read_file("E:\\Data\\06-09_2000-2021\\nwh_shape\\Export_Output.shp")
#in_states = states.query("admin = 'INDIA'").reset_index(drop = True)
state_ids = {k: i for i, k in enumerate(states.Name)}
shapes = [(shape, n) for n, shape in enumerate(states.geometry)]
############### Reading Data File ################################
############### Reading Data File ################################


ds = xr.open_mfdataset("E:\\Data\\06-09_2000-2021\\*.nc")
rf = ds.precipitationCal.values

clim = np.nanmean(rf, axis = 0)

lon = ds.lon.values
lat = ds.lat.values
############# Rasterizing data ###########################
ds['states'] = rasterize(shapes,ds.coords)
jk =ds.states.where(ds.states == state_ids['Jammu and Kashmir'])
jnk =np.where(jk ==0,1, jk)
###########################################################
climatology = (clim*jnk.T).T

import cmaps
states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='110m',
        facecolor='none')

central_lon, central_lat = 77.5,33.5
extent = [72.25, 81,28,37.2]

scale = '110m'
fig = plt.figure(figsize=(5,5))
ax = plt.axes(projection=ccrs.PlateCarree())
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
ax.set_global()
ax.set_extent(extent)
df = gpd.read_file('E:\\Data\\06-09_2000-2021\\state_shape\\India_State_Boundary.shp')
levels = [0.1,0.5,1,1.5,2,3,4,5,6,7,8,9,10,11,12,13]
cmap = plt.get_cmap('jet')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
cp =plt.pcolormesh(lon,lat,climatology,transform=ccrs.PlateCarree(),cmap =truncate_colormap(cmaps.BlueDarkRed18, 0.2, 0.9), norm = norm)
plotly =df.loc[(df['Name'] == 'Uttarakhand') | (df['Name'] == 'Jammu and Kashmir') | (df['Name'] =='Himachal Pradesh') | (df['Name'] =='Ladakh')].plot(ax=ax, edgecolor='black',linewidth = 2, facecolor ='None')

ax.set_xticks([74,76,78,80], crs=ccrs.PlateCarree())
ax.set_yticks([30,32,34,36], crs=ccrs.PlateCarree())
ax.coastlines(scale)
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
pos =ax.get_position()
plt.xlabel('Longitude', fontsize ='16', weight = 'bold')
plt.ylabel('Latitude', fontsize ='16', weight = 'bold')

cb_ax1 = fig.add_axes([pos.x1+0.025,pos.y0,0.025,0.76])
cb5 =plt.colorbar(cp,cax = cb_ax1, orientation='vertical',ticks=[0.1,0.5,1,1.5,2,3,4,5,6,7,8,9,10,11,12,13], drawedges=True)

fig1 = plt.gcf()
fig1.savefig('E:Data/Results/Regrid_result/IMD_climatology_cmap.pdf', dpi =600)

fig1.savefig('E:Data/Results/Regrid_result/IMD_climatology_cmap.png', dpi=600)
plt.show()



