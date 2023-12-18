# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 14:02:23 2023
@author: danie

Compute historical yearly growing season GDDs and cumulative precipitation on a 0.5x0.5 grid over Europe using ISMIP2b agro-climatic data.

"""

import xarray as xr
import numpy as np
import time
import cftime
#%% 1. Loading the Input Variables, the raw agro-clim. data.
#1a. Write down the file paths.
precip_path = "Bioclim_vars/pr_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_europe_18610101-20051231.nc4"
tasmax_path = "Bioclim_vars/tasmax_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_europe_18610101-20051231.nc4"
tasmin_path = "Bioclim_vars/tasmin_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_europe_18610101-20051231.nc4"
plant_day_path = "Bioclim_vars/gepic_hadgem2-es_ewembi_historical_2005soc_co2_plantday-soy-noirr_lat36.25to70.25lon-11.25to40.25_annual_1861_2005.nc4"
mature_day_path = "Bioclim_vars/gepic_hadgem2-es_ewembi_historical_2005soc_co2_matyday-soy-noirr_lat36.25to70.25lon-11.25to40.25_annual_1861_2005.nc4"

#1b. Load in the files using xarray, and save them as numpy arrays.
tasmax = xr.open_dataset(tasmax_path,decode_times=False)['tasmax'].values
tasmin= xr.open_dataset(tasmin_path,decode_times=False)['tasmin'].values
precip= xr.open_dataset(precip_path,decode_times=False)['pr'].values
plant_day = xr.open_dataset(plant_day_path,decode_times=False)['plantday-soy-noirr'].values
mature_day = xr.open_dataset(mature_day_path,decode_times=False)['matyday-soy-noirr'].values

#%% 2. Creating files for holding the output variables.   The variables are GDDs and cum. precip, the files will have the same dimensions as the input data which are yearly.
GDD_frame = xr.open_dataset(plant_day_path,decode_times=False)
GDD_frame = GDD_frame.assign(GDD=GDD_frame["plantday-soy-noirr"]*0)
GDD_frame = GDD_frame.drop_vars('plantday-soy-noirr')

Precip_frame = xr.open_dataset(plant_day_path,decode_times=False)
Precip_frame = Precip_frame.assign(pre=Precip_frame["plantday-soy-noirr"]*0)
Precip_frame = Precip_frame.drop_vars('plantday-soy-noirr')

#%% 3. Compute the output variables
#3a. Define useful functions for computing growing season GDDs & Cumul. Precip. 
def CheckLeap(Year):  
  # Checking if the given year is leap year  
  if((Year % 400 == 0) or  
     (Year % 100 != 0) and  
     (Year % 4 == 0)):   
    return True;
  else:
    return False;  
def DaysBefore(Year):  
  relative_to_year = 1861
  days = 0
  if Year != 0:
      for year in np.arange(relative_to_year,relative_to_year+Year):
          if CheckLeap(year) == True:
              days = days + 366
          if CheckLeap(year) == False:
              days = days + 365
  return days

timer_start = time.time()
#3b. computing GDDs
#i. Find the lat/lon of unmasked values of GDD
lats,lons= np.where(np.sum(plant_day,axis=0)>0)
years = np.arange(0,145,1)
#ii. Find planting day timeseries at each lat/lon
yearstart = [DaysBefore(year) for year in years]
planting = [yearstart+plant_day[:,i,j] for i,j  in zip(lats,lons) ] 
maturation = [yearstart+mature_day[:,i,j] for i,j  in zip(lats,lons) ] 
timer_end = time.time()
#iii. Compute GDD's using list comprehension. -- 
soy_min = 10;
soy_max = 40;
GDD= [[ np.sum(np.clip(
                   ((tasmax[int(start):int(end),lats[i],lons[i]]+tasmin[int(start):int(end),lats[i],lons[i]])/2)-273.15-10,0,soy_max-soy_min)) 
                        for start, end in zip(planting[i],maturation[i])  ]
                          for i in np.arange(0,4248)]

#3c. Computing growing season cum. precipitation
precipitation= [[ np.sum(precip[int(start):int(end),lats[i],lons[i]]) for start, end in zip(planting[i],maturation[i])]  for i in np.arange(0,4248)]
#%% 4. Fill the output files with the output variables.
for i in np.arange(0,len(lons)):
    GDD_frame["GDD"][:,lats[i],lons[i]] = GDD[i]
    Precip_frame["pre"][:,lats[i],lons[i]] = precipitation[i]
    
#%% 5. Save the output files.
GDD_frame.to_netcdf('./Bioclim_vars/gepic_hadgem2-es_ewembi_historical_2005soc_co2_GDD_growing_season-soy-noirr.nc4')
Precip_frame.to_netcdf('./Bioclim_vars/pr_cumul_growing_season_HadGEM2-ES_historical_r1i1p1_EWEMBI_europe_18610101-20051231.nc4')
