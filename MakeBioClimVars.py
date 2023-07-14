# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 21:39:48 2023
@author: danie

#
Construct pertinent yearly bio-climatic indicators using daily climatic data. 
#
"""

import numpy as np
import netCDF4 
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import time
#%%1. Make an empty NetCDF container for the bioclim. variables
def nc_shell(file_name,variable,units):
    fn = file_name
    ds = netCDF4.Dataset(fn, 'w', format='NETCDF4')
    # def nc_shell(ds)
    time = ds.createDimension('time', 145)
    lat = ds.createDimension('lat', 69)
    lon = ds.createDimension('lon', 104)
    
    times = ds.createVariable('time', 'f4', ('time',))
    lats = ds.createVariable('lat', 'f4', ('lat',))
    lons = ds.createVariable('lon', 'f4', ('lon',))
    value = ds.createVariable(variable, 'f4', ('time', 'lat', 'lon',))
    value.units = units
    times.units = 'years since 1661-1-1 00:00:00'
    times[:] = np.arange(200, 344+1, 1)
    lats[:] = np.arange(36.25, 70.25+0.5, 0.5)
    lons[:] = np.arange(-11.25,40.25+0.5,0.5)
    return ds
#HTC = nc_shell('./Bioclim_vars/gepic_hadgem2-es_ewembi_historical_2005soc_co2_HTC_growing_season-soy-noirr.nc4','Hydrothermal_Coeff','mm/Kelvin')
GDD = nc_shell('./Bioclim_vars/gepic_hadgem2-es_ewembi_historical_2005soc_co2_GDD_growing_season-soy-noirr.nc4','Growing_Degree_Days','Kelvin')
Precip = nc_shell('./Bioclim_vars/pr_cumul_growing_season_HadGEM2-ES_historical_r1i1p1_EWEMBI_europe_18610101-20051231.nc4','Cumul_Precip','mm')
#%%2. Load in the raw bioclim. variable ingredients
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

#%%3. Make Growing Degree Days and cum. precipitation in a for loop (takes roughly 2.6 hours)

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

              
start=time.time()
for i in np.arange(0,len(lat)):
    for j in np.arange(0,len(lon)):
        for year in np.arange(0,len(plant_day)):
            if plant_day[year,i,j].mask == False:
                soy_min = 10;
                soy_max = 40;
                planting = int(DaysBefore(year)+plant_day[year,i,j])
                maturation = int(planting+mature_day[year,i,j])
                mean_t = (tasmax[planting:maturation,i,j]+tasmin[planting:maturation,i,j])/2
                P = sum(precip[planting:maturation,i,j])*86400
                # active_t =sum((mean_t-273.15)[mean_t-273.15>10])*10
                # if active_t>0:
                #     H = P/sum((mean_t-273.15)[mean_t-273.15>10])*10
                # else:
                #     H = 0
                G =  sum(np.clip(mean_t-273.15-10,0,soy_max-soy_min))
                if np.isnan(G):
                    print('nan!')
                    print((i,j))
                GDD['Growing_Degree_Days'][year,i,j]=G
                Precip['Cumul_Precip'][year,i,j]=P
                #HTC['Hydrothermal_Coeff'][year,i,j]=H
            else:
                GDD['Growing_Degree_Days'][year,i,j] = None
                Precip['Cumul_Precip'][year,i,j] = None
                #HTC['Hydrothermal_Coeff'][year,i,j]= None
    print(i)
end=time.time()
GDD.close()
Precip.close()
#HTC.close()



