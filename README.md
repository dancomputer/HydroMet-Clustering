# HydroMet-Clustering, v1a

## Code
**Cluster.py** is the main code. It first defines bio-climatic regions, then yield-time-series clusters within regions.

**MakeBioClimVars.py** calculates GDDs, cumulative precipitation, and (unused) hydrothermal coefficient. **MakeClusteredResultNC4.py** makes the clustered data netcdf4 file.

**DaysBefore.py** y = DaysBefore(x) is a 1-D map from x = year relative to 1861 to y = days relative to 1861. 
i.e. x = 0 means 1861, and so y=0; x =1 means 1862, and so y = 365, and so on, 
accounting for leap years!

## Data
The results of _Cluster.py_ is 9 regions containing 805 clusters in total. These clusters are marked as xxxx where the first digit is the region and the last three are the cluster number within the region. 

The clustered data, 'Clustered_Data_v1a.nc4', is uploaded to the [_CropWeather_Risk_ GDrive](https://drive.google.com/drive/u/1/folders/1mB1umEvFzYN4-NWyQZRF-QI8yr9iBoNv). For each pixel, the cluster and the cluster-averaged data are accesible. Precipitation and Max Temperatures are given daily for 52960 days, while the other time-series data is given yearly for 145 years. To index days based on years, use _DaysBefore.py_.  Averaged planting/maturity day can take non-integer values so might need to be rounded.





