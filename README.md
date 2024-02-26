# HydroMet-Clustering, v1a

## Code
**Cluster.py** is the main code. It first defines agro-climatic regions, then yield-time-series clusters within regions.

**MakeBioClimVars.py** calculates GDDs, cumulative precipitation, and (unused) hydrothermal coefficient.

**MakeClusteredResultNC4.py** makes the clustered data netcdf4 file.



## Data
Clusters are marked as xxxx where the first digit is the region and the last three are the cluster number within the region. 

The results 'Clustered_Data_v1a.nc4' are uploaded to the [_CropWeather_Risk_ GDrive](https://drive.google.com/drive/u/1/folders/1mB1umEvFzYN4-NWyQZRF-QI8yr9iBoNv). 

To index days based on years, use DaysBefore() from MakeBioClimVars.py. Averaged planting/maturity day can take non-integer values so might need to be rounded.

The conda environemnt with compatible package versions is named "HydroMet-Clustering_PythonEnvironment.yml". 




