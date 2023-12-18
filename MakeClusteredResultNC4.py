# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 16:14:21 2023

@author: danie

This code aggregates data based on the clusters found in Cluster.py, and writes it into an NC4 file. 
It just requires the final Geopandas frame 'gdf' from Cluster.py.

"""
import numpy as np
import netCDF4
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

#%%1. Load in the raw bioclim. variable ingredients
#1a. Chunking
# Define the chunk sizes for raw daily variables
pd.options.mode.chained_assignment = None  # default='warn'
#1b. Load read netcdf4 data files
yield_path="./Bioclim_vars\gepic_hadgem2-es_ewembi_historical_2005soc_co2_yield-soy-noirr_lat36.25to70.25lon-11.25to40.25_annual_1861_2005.nc4";
precip_path = "Bioclim_vars/pr_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_europe_18610101-20051231.nc4"
tasmax_path = "Bioclim_vars/tasmax_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_europe_18610101-20051231.nc4"
tasmin_path = "Bioclim_vars/tasmin_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_europe_18610101-20051231.nc4"
plant_day_path = "Bioclim_vars/gepic_hadgem2-es_ewembi_historical_2005soc_co2_plantday-soy-noirr_lat36.25to70.25lon-11.25to40.25_annual_1861_2005.nc4"
mature_day_path = "Bioclim_vars/gepic_hadgem2-es_ewembi_historical_2005soc_co2_matyday-soy-noirr_lat36.25to70.25lon-11.25to40.25_annual_1861_2005.nc4"
# Load in the raw bioclim. variable ingredients using xarray
soy_yield = xr.open_dataset(yield_path,decode_times=False)['yield-soy-noirr'].values
tasmax = xr.open_dataset(tasmax_path,decode_times=False)['tasmax'].values
tasmin= xr.open_dataset(tasmin_path,decode_times=False)['tasmin'].values
precip= xr.open_dataset(precip_path,decode_times=False)['pr'].values
plant_day = xr.open_dataset(plant_day_path,decode_times=False)['plantday-soy-noirr'].values
mature_day = xr.open_dataset(mature_day_path,decode_times=False)['matyday-soy-noirr'].values

#%%2. Make nc4 file shell
def nc_shell():
    ds = xr.Dataset()

    ds['time'] = np.arange(0, 52960, 1)
    ds['year'] = np.arange(0, 145)
    ds['lat'] = np.arange(70.25, 36.25 - 0.5, -0.5)
    ds['lon'] = np.arange(-11.25, 40.25 + 0.5, 0.5)

    ds['Cluster'] = (('lat', 'lon'), np.zeros((len(ds['lat']), len(ds['lon']))))
    ds['Pl_day'] = (('year', 'lat', 'lon'), np.zeros((len(ds['year']), len(ds['lat']), len(ds['lon']))))
    ds['Mat_day'] = (('year', 'lat', 'lon'), np.zeros((len(ds['year']), len(ds['lat']), len(ds['lon']))))
    ds['Yield'] = (('year', 'lat', 'lon'), np.zeros((len(ds['year']), len(ds['lat']), len(ds['lon']))))
    ds['Precip'] = (('time', 'lat', 'lon'), np.zeros((len(ds['time']), len(ds['lat']), len(ds['lon']))))
    ds['Tmax'] = (('time', 'lat', 'lon'), np.zeros((len(ds['time']), len(ds['lat']), len(ds['lon']))))

    ds['year'].attrs['units'] = 'years since 1861-1-1 00:00:00'
    ds['time'].attrs['units'] = 'days since 1861-1-1 00:00:00'
    ds['Pl_day'].attrs['units'] = 'day of year, cluster avg.'
    ds['Mat_day'].attrs['units'] = 'day of year, cluster avg.'
    ds['Yield'].attrs['units'] = 't ha-1 yr-1, cluster avg.'
    ds['Precip'].attrs['units'] = 'kg m-2 s-1, cluster avg.'
    ds['Tmax'].attrs['units'] = 'K, cluster avg.'
    return ds

Final_Result_Frame=nc_shell()
#%% 3a. Aggregate data by cluster.
clustered_yields =  [np.mean([soy_yield[:,pair[0],pair[1]]  for pair in gdf[gdf['ts_labels']==cluster]['lat-lon']]) for cluster in gdf['ts_labels'].unique()]  
clustered_planting =  [np.mean([plant_day[:,pair[0],pair[1]]  for pair in gdf[gdf['ts_labels']==cluster]['lat-lon']]) for cluster in gdf['ts_labels'].unique()]  
clustered_maturity =  [np.mean([mature_day[:,pair[0],pair[1]]  for pair in gdf[gdf['ts_labels']==cluster]['lat-lon']]) for cluster in gdf['ts_labels'].unique()]  
clustered_precip =  [np.mean([precip[:,pair[0],pair[1]]  for pair in gdf[gdf['ts_labels']==cluster]['lat-lon']]) for cluster in gdf['ts_labels'].unique()]  
clustered_tasmax =  [np.mean([tasmax[:,pair[0],pair[1]]  for pair in gdf[gdf['ts_labels']==cluster]['lat-lon']]) for cluster in gdf['ts_labels'].unique()]  

#%%4. Fill file with aggregate data.
latlon = gdf['lat-lon'].values
labels = gdf['ts_labels'].values
label_to_ind = {gdf['ts_labels'].unique()[i]:int(i) for i in np.arange(0,len(gdf['ts_labels'].unique()),1) }
for pixel in np.arange(0,len(gdf['lat-lon'])):
    print(pixel)
    Final_Result_Frame['Cluster'][latlon[pixel][0],latlon[pixel][1]]=labels[pixel]
    ind = label_to_ind[labels[pixel]]
    Final_Result_Frame['Yield'][:,latlon[pixel][0],latlon[pixel][1]] = clustered_yields[ind]
    Final_Result_Frame['Pl_day'][:,latlon[pixel][0],latlon[pixel][1]] = clustered_planting[ind]
    Final_Result_Frame['Mat_day'][:,latlon[pixel][0],latlon[pixel][1]] = clustered_maturity[ind]
    Final_Result_Frame['Precip'][:,latlon[pixel][0],latlon[pixel][1]] = clustered_precip[ind]
    Final_Result_Frame['Tmax'][:,latlon[pixel][0],latlon[pixel][1]] = clustered_tasmax[ind]
#Save file
Final_Result_Frame.to_netcdf('./Clustering_results_v1a.nc4')

