# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 16:14:21 2023

@author: danie

This code aggregates data based on the clusters found in Cluster.py, and writes it into an NC4 file. 
It just requires the final Geopandas frame 'gdf' from Cluster.py.

Currently the output NC4 file is named test.nc4 . Simply change this dummy name to a more useful name after this code is finished running.

Warning: It is very slow -- will take about 24 hours.
"""

import numpy as np
import netCDF4 
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import time

#%%1. Load in the raw bioclim. variable ingredients
import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

pd.options.mode.chained_assignment = None  # default='warn'
plant_path = "C:/Users/danie/OneDrive/Desktop/Giannini+Ghil+Chavez+Ohara/gepic_hadgem2-es_ewembi_historical_2005soc_co2_plantday-soy-noirr_lat36.25to70.25lon-11.25to40.25_annual_1861_2005.nc4"
yield_path="C:/Users\danie\OneDrive\Desktop\Giannini+Ghil+Chavez+Ohara\Bioclim_vars\gepic_hadgem2-es_ewembi_historical_2005soc_co2_yield-soy-noirr_lat36.25to70.25lon-11.25to40.25_annual_1861_2005.nc4";
pixel_map_path="C:/Users/danie/OneDrive/Desktop/Giannini+Ghil+Chavez+Ohara/Europe-mai-large-gepic/nhmm-data/pixel-map_gepic_hadgem2-es_ewembi_historical_2005soc_co2_yield-soy-noirr_europe.txt";

f = netCDF4.Dataset(yield_path)
soy_yield = f['yield-soy-noirr']
time=f.variables['time']
yield_lat=f.variables['lat']
yield_lon=f.variables['lon']
yield_path="C:/Users\danie\OneDrive\Desktop\Giannini+Ghil+Chavez+Ohara\Bioclim_vars\gepic_hadgem2-es_ewembi_historical_2005soc_co2_yield-soy-noirr_lat36.25to70.25lon-11.25to40.25_annual_1861_2005.nc4";
soy_yield = netCDF4.Dataset(yield_path)['yield-soy-noirr']

precip_path = "Bioclim_vars/pr_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_europe_18610101-20051231.nc4"
tasmax_path = "Bioclim_vars/tasmax_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_europe_18610101-20051231.nc4"
tasmin_path = "Bioclim_vars/tasmin_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_europe_18610101-20051231.nc4"
plant_day_path = "Bioclim_vars/gepic_hadgem2-es_ewembi_historical_2005soc_co2_plantday-soy-noirr_lat36.25to70.25lon-11.25to40.25_annual_1861_2005.nc4"
mature_day_path = "Bioclim_vars/gepic_hadgem2-es_ewembi_historical_2005soc_co2_matyday-soy-noirr_lat36.25to70.25lon-11.25to40.25_annual_1861_2005.nc4"

tasmax = netCDF4.Dataset(tasmax_path)['tasmax']
tasmin = netCDF4.Dataset(tasmin_path)['tasmin']
precip = netCDF4.Dataset(precip_path)['pr']
plant_day = netCDF4.Dataset(plant_day_path)['plantday-soy-noirr']
mature_day = netCDF4.Dataset(mature_day_path)['matyday-soy-noirr']

lat=netCDF4.Dataset(tasmax_path)['lat']
lon=netCDF4.Dataset(tasmax_path)['lon']
#%%2. Make nc4 file shell
def nc_shell(file_name):
    fn = file_name
    ds = netCDF4.Dataset(fn, 'w', format='NETCDF4')
    time = ds.createDimension('time', 52960)
    year = ds.createDimension('year',145)
    lat = ds.createDimension('lat', 69)
    lon = ds.createDimension('lon', 104)
    
    times = ds.createVariable('time', 'f4', ('time',))
    years = ds.createVariable('year', 'f4', ('year',))
    lats = ds.createVariable('lat', 'f4', ('lat',))
    lons = ds.createVariable('lon', 'f4', ('lon',))
    
    cluster = ds.createVariable('Cluster', 'f4', ('lat', 'lon',))
    
    plant_day=ds.createVariable('Pl_day', 'f4', ('year','lat', 'lon',))
    mat_day=ds.createVariable('Mat_day', 'f4', ('year','lat', 'lon',))
    yields = ds.createVariable('Yield', 'f4', ('year','lat', 'lon',))
    precip = ds.createVariable('Precip', 'f4', ('time', 'lat', 'lon',))
    tmax = ds.createVariable('Tmax', 'f4', ('time', 'lat', 'lon',))
    
    years.units = 'years since 1861-1-1 00:00:00'
    times.units = 'days since 1861-1-1 00:00:00'
    plant_day.units = 'day of year'
    mat_day.units = 'day of year'
    yields.units = 't ha-1 yr-1'
    precip.units = 'kg m-2 s-1'
    tmax.units ='K'
    
    times[:] = np.arange(0, 52960, 1)
    years[:] = np.arange(0,145)
    lats[:] = np.arange(36.25, 70.25+0.5, 0.5)
    lons[:] = np.arange(-11.25,40.25+0.5,0.5)
    return ds

test=nc_shell('HydroMet_Clustering_v1a.nc4')

#%% 3a. Aggregate data by cluster.
start=time.time()
clustered_yields={i:np.zeros(145) for i in gdf['ts_labels'].unique()}
clustered_planting={i:np.zeros(145) for i in gdf['ts_labels'].unique()}
clustered_maturity={i:np.zeros(145) for i in gdf['ts_labels'].unique()}
clustered_precip={i:np.zeros(52960) for i in gdf['ts_labels'].unique()}
clustered_tmax={i:np.zeros(52960) for i in gdf['ts_labels'].unique()}
for cluster in gdf['ts_labels'].unique():
    print('now aggregating for cluster ' + str(cluster))
    for pair in gdf[gdf['ts_labels']==cluster]['pair']:
        cluster_length=len(gdf[gdf['ts_labels']==cluster]['pair'])
        clustered_yields[cluster] = clustered_yields[cluster] + soy_yield[:,pair[0],pair[1]]/cluster_length
        clustered_planting[cluster] = clustered_planting[cluster] + plant_day[:,pair[0],pair[1]]/cluster_length
        clustered_maturity[cluster] = clustered_maturity[cluster] + mature_day[:,pair[0],pair[1]]/cluster_length
        clustered_precip[cluster] = clustered_precip[cluster] + precip[:,pair[0],pair[1]]/cluster_length
        clustered_tmax[cluster] = clustered_tmax[cluster] + tasmax[:,pair[0],pair[1]]/cluster_length
end1=time.time()

#%% 3b. Fill nc4 frame.
start=time.time()
count = 0
be=0
for i in np.arange(0,len(lat)):
    for j in np.arange(0,len(lon)):
        if gdf[gdf['count']==count].empty == False:
            be=be+1 
            label = int(gdf[gdf['count']==count]['ts_labels'])
            test['Cluster'][i,j] = label
            test['Yield'][:,i,j] = clustered_yields[label]
            test['Pl_day'][:,i,j] = clustered_planting[label]
            test['Mat_day'][:,i,j] = clustered_maturity[label]
            test['Precip'][:,i,j] = clustered_precip[label]
            test['Tmax'][:,i,j] = clustered_tmax[label]
        else:
            test['Cluster'][i,j] = np.ma.array(float(0),mask=1)
            test['Yield'][:,i,j] = np.ma.array(np.zeros(145), mask=np.ones(145))
            test['Pl_day'][:,i,j] = np.ma.array(np.zeros(145), mask=np.ones(145))
            test['Mat_day'][:,i,j] = np.ma.array(np.zeros(145), mask=np.ones(145))
            test['Precip'][:,i,j] = np.ma.array(np.zeros(52960), mask=np.ones(52960))
            test['Tmax'][:,i,j] = np.ma.array(np.zeros(52960), mask=np.ones(52960))
        count= count+1
    print(i)
end2=time.time()
test.close()
