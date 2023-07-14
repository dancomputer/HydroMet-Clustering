# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 09:43:39 2023

@author: danie

#
Regionalize by bio-climatic statistics, then cluster time series within these regions.
#
"""
#%%0. Load in data

import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

#Yield 
pd.options.mode.chained_assignment = None  # default='warn'
plant_path = "C:/Users/danie/OneDrive/Desktop/Giannini+Ghil+Chavez+Ohara/gepic_hadgem2-es_ewembi_historical_2005soc_co2_plantday-soy-noirr_lat36.25to70.25lon-11.25to40.25_annual_1861_2005.nc4"
yield_path="C:/Users\danie\OneDrive\Desktop\Giannini+Ghil+Chavez+Ohara\Bioclim_vars\gepic_hadgem2-es_ewembi_historical_2005soc_co2_yield-soy-noirr_lat36.25to70.25lon-11.25to40.25_annual_1861_2005.nc4";
pixel_map_path="C:/Users/danie/OneDrive/Desktop/Giannini+Ghil+Chavez+Ohara/Europe-mai-large-gepic/nhmm-data/pixel-map_gepic_hadgem2-es_ewembi_historical_2005soc_co2_yield-soy-noirr_europe.txt";

f = netCDF4.Dataset(yield_path)
soy_yield = f['yield-soy-noirr']
time=f.variables['time']
yield_lat=f.variables['lat']
yield_lon=f.variables['lon']

#bioclim. variables
GDD_path = "Bioclim_vars/gepic_hadgem2-es_ewembi_historical_2005soc_co2_GDD_growing_season-soy-noirr.nc4"
GDD = netCDF4.Dataset(GDD_path)['Growing_Degree_Days']
Pre_path = './Bioclim_vars/pr_cumul_growing_season_HadGEM2-ES_historical_r1i1p1_EWEMBI_europe_18610101-20051231.nc4'
Pre = netCDF4.Dataset(Pre_path)['Cumul_Precip']

#raw bioclim. variable ingredients
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

#%%1. Get data into a structured setting
#%%1a. Bio-Clim stats
Bio_Clim_variables=['GDD_mean','Pre_mean','GDD_var','Pre_var','yield_mean','yield_std','yield_max','yield_min']
count = 0;
a = [None]*(69*104)
b={}
for i in np.arange(0,len(yield_lat)):
    for j in np.arange(0,len(yield_lon)):
        soy_slice = soy_yield[:,i,j]
        if (soy_slice.mask.sum() == 0) & (GDD[:,i,j].mask.sum()==0):  
            a[count] = (np.double(GDD[:,i,j].mean()),
                        np.double(Pre[:,i,j].mean()),
                        np.double(GDD[:,i,j].var()),
                        np.double(Pre[:,i,j].var()),
                        soy_slice.data.mean(),
                        soy_slice.data.std(),
                        soy_slice.data.max(),
                        soy_slice.data.min(),
                        )
            b.update({count:(i,j)})
            count = count + 1
        else: 
            a[count] = (np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)
            count = count + 1
    print(i)
print(count)
Bio_Clim_Stats=pd.DataFrame(a,columns=Bio_Clim_variables)

#%%1b. Bio-Clim time series 
ts_variables=['Yield','pair'] #raw ingerdients: not neccesary, currently. 'Raw_tasmax','Raw_pre','Raw_plantday','Raw_matday'
count = 0;
a = [None]*(69*104)
for i in np.arange(0,len(yield_lat)):
    for j in np.arange(0,len(yield_lon)):
        soy_slice = soy_yield[:,i,j]
        pair = (i,j)
        if (soy_slice.mask.sum() == 0) & (GDD[:,i,j].mask.sum()==0):  
                a[count] = (soy_slice.data,
                            pair,
                            count)
                b.update({count:(i,j)})
                count = count + 1
        else: 
            a[count] = (np.nan,pair,count)
            count = count + 1
    print(i)
print(count)
Temporal_Variables = pd.DataFrame(a,columns=ts_variables+['count'])
#%%2. Set up dataframe 

xm,ym = np.meshgrid(yield_lon[:],yield_lat[:]);
data = {'lon':xm.flatten(),'lat':ym.flatten(),}
coordinates = pd.DataFrame.from_dict(data);
df = pd.concat([coordinates,Temporal_Variables,Bio_Clim_Stats],axis=1)

#%%. Spatially plot some chosen mean statistics 
cluster_variables=['GDD_mean','Pre_mean','yield_mean']
f, axs = plt.subplots(nrows=len(cluster_variables), ncols=1, figsize=(12, 12))
# Make the axes accessible with single indexing
axs = axs.flatten()
# Start a loop over all the variables of interest
for i, col in enumerate(cluster_variables):
    # select the axis where the map will go
    ax = axs[i]
    # Plot the map
    df.plot(x="lon", y="lat", kind="scatter", c=col, colormap="YlOrRd", ax=ax)
    # Remove axis clutter
    ax.set_axis_off()
    # Set the axis title to the name of variable being plotted
    ax.set_title(col)
# Display the figure
plt.show()
#%%3. Transform into geopandas set
from libpysal.weights import Queen, KNN
import geopandas
import shapely 
#Slice away all NA datapoints
cut_df = df[df[ts_variables[0]].notna()]
gdf_points = geopandas.GeoDataFrame(cut_df, 
            geometry=geopandas.points_from_xy(cut_df.lon, cut_df.lat),
            crs="EPSG:4326")

def get_square_around_point(point_geom, delta_size=0.25):
    
    point_coords = np.array(point_geom.coords[0])

    c1 = point_coords + [-delta_size,-delta_size]
    c2 = point_coords + [-delta_size,+delta_size]
    c3 = point_coords + [+delta_size,+delta_size]
    c4 = point_coords + [+delta_size,-delta_size]
    
    square_geom = shapely.geometry.Polygon([c1,c2,c3,c4])
    
    return square_geom

def get_gdf_with_squares(gdf_with_points, delta_size=0.25):
    gdf_squares = gdf_with_points.copy()
    gdf_squares['geometry'] = (gdf_with_points['geometry']
                               .apply(get_square_around_point, 
                                      delta_size))
    
    return gdf_squares

gdf = get_gdf_with_squares(gdf_points, delta_size=0.25)
#%%4. Define a distance matrix for regionalization
from esda.moran import Moran
import numpy
w = Queen.from_dataframe(gdf_points)
#%%5. Scale the bio-clim statistics using a robust scale. Robust scaling minimizes the effects of outliers.
import sklearn
from sklearn.preprocessing import robust_scale, scale, minmax_scale
bioclim_stats = ['GDD_mean','Pre_mean','yield_mean']
scaled_stats = robust_scale(gdf[bioclim_stats]) #scales each column

#extra 
gdf['GDD_mean_scaled']=robust_scale(gdf['GDD_mean'])
gdf['Pre_mean_scaled']=robust_scale(gdf['Pre_mean'])
gdf['yield_mean_scaled']=robust_scale(gdf['yield_mean'])
#%%6a. Regionalization (Thanks to an inital inspiration by code from Badr Moufad)
#%%6a.i. Identify good number of regions
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
metric = 'euclidean'
linkage_method = 'ward'
X = scaled_stats
Z = linkage(X, metric=metric, method=linkage_method)

#Check the clustering dendrogram -- an additional, manual heurisitic for determing ideal number of regions.
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dn = dendrogram(Z,
                truncate_mode='lastp',  # show only the last p merged clusters
                p=24,  # show only the last p merged clusters
                leaf_rotation=90.,
                leaf_font_size=12.,
                show_contracted=True,  # to get a distribution impression in truncated branches
                )

#Choose the good cluster number, evaluated by their clustering scores. 
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
arr_score = {}
# max possible number of regions
max_k = 30
# loop over number of regions
for k in range(2, max_k):
  # build model 
  model = AgglomerativeClustering(n_clusters=k,metric=metric,linkage=linkage_method,connectivity=w.sparse)

  model = model.fit(X)

  # regions label
  labels = model.fit_predict(X)

  # compute score metric
  #m = metrics.calinski_harabasz_score(X, labels)
  m = metrics.silhouette_score(X, labels)
  #m = metrics.davies_bouldin_score(X,labels)
  arr_score[k] = m

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(list(arr_score.keys()),list(arr_score.values()))
ax.set_xlabel('# clusters')
ax.set_ylabel('score')
plt.show()

### now, choose the number of regions before a big drop off in score.
best_n_regions = max(arr_score, key=arr_score.get)
print(best_n_regions)
best_n_regions=int(input('Set n regions. Calculation gave ' + str(best_n_regions)+ ' as optimal.: '))
#%%6a.ii Agg. clustering with good number of regions.
model = AgglomerativeClustering(n_clusters=best_n_regions,metric=metric,linkage=linkage_method,connectivity=w.sparse)
model = model.fit(X)

# clusters label
gdf["unsorted_region"] =  model.fit_predict(X)
gdf["region"] = np.zeros(4248)
from pylab import cm
import matplotlib.colors as mcolors
import matplotlib.ticker as tkr 

#Order regions by mean yield. With such an ordering, the rainbow color progression signifies increasing mean regional yield.
regional_yields={i: (gdf[gdf['unsorted_region']==i]['yield_mean'].sum())/len(gdf[gdf['unsorted_region']==0]) for i in gdf['unsorted_region'].unique()}
for index,region in enumerate(list(dict(sorted(regional_yields.items(), key=lambda item: item[1])).keys())):
    gdf.loc[gdf.unsorted_region == region,'region'] = index

gdf.drop('unsorted_region',axis=1)
###Plotting regions
fig, ax = plt.subplots(figsize=(8,6))
# plot map on axis
countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
countries.plot(color="lightgrey",ax=ax)
# plot points

cmap = cm.get_cmap('gist_rainbow', best_n_regions)
color_list = [mcolors.rgb2hex(cmap(i)) for i in range(cmap.N)]
plot = gdf.plot(x="lon", y="lat", kind="scatter", c='region', colormap=cmap, ax=ax, legend=True)
# add grid
ax.set_xlim([min(yield_lon),max(yield_lon)])
ax.set_ylim([min(yield_lat),max(yield_lat)])
plt.title("distance: " + metric + "; " + "variables: " + ", ".join(cluster_stats))
#ax.grid(b=True, alpha=0.5)
#
a= plt.show()


#%%Visualize regional stats
x,y,z=gdf['GDD_mean'],gdf['Pre_mean'],gdf['yield_mean']
import plotly.graph_objects as go
import plotly.express as px
import plotly as py

gdf['region_str']=gdf['region'].astype(str)
fig = px.scatter_3d(gdf, x="GDD_mean_scale", y="Pre_mean_scale", z="yield_mean_scale",color="region_str",color_discrete_sequence=color_list,
                    category_orders={"region_str": list(np.arange(0,9).astype(str))})
fig.update_traces(marker=dict(size=2))
fig.show()
#py.offline.plot(fig, filename = './Plots/v1a_Regions_Yield_3DScatter.html')

#%%7a. Prepare time-series data for clustering 
cluster_ts = ['Yield']
gdf_scaled_variables = np.ndarray((len(gdf),145,len(cluster_ts)))
# packed_data = test['scaled_Pre'].copy()
# for key, d in list(test.items())[1:]:
#   packed_data = packed_data.join(d, rsuffix=f"_{key}") 

for i in np.arange(0,len(gdf),1):
    for j in np.arange(len(cluster_ts)):
        print(i)
        gdf_scaled_variables[i,:,j]= scale(gdf[cluster_ts[j]].iloc[i][:])
          
#%%7b.i. Identify good number of clusters
from matplotlib import pyplot as plt
from tslearn.clustering import TimeSeriesKMeans, silhouette_score

gdf['ts_labels']=gdf['region']*1000 
for region in np.arange(0,best_n_regions,1):
    region_index = np.where(gdf['region']==region)
    X=gdf_scaled_variables[region_index]
    print('clustering on region ' + str(region) + '. Total # pixels: ' + str(X.shape[0]))
    max_k = round(X.shape[0]/5) #This very roughly limits the minimal cluster size to 5.
    
    arr_score = {}
    if max_k>1:
        for k in range(2,max_k+1):
          # build model 
          model = TimeSeriesKMeans(n_clusters=k, metric=metric,
                                   max_iter=50, random_state=10)
          model.fit(X)
          labels = model.labels_
        
          # compute score metric
          #m = metrics.calinski_harabasz_score(X, labels)
          #m = metrics.silhouette_score(X, labels)
          m = silhouette_score(X,labels,metric=metric)
          arr_score[k] = m
        
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.bar(list(arr_score.keys()),list(arr_score.values()))
        ax.set_xlabel('# clusters')
        ax.set_ylabel('score')
        plt.title('Cluster scores for region '+ str(region))
        plt.show()
        
        ###Here, simply choose the number of clusters with the highest score. We just want to cluster very similar time series, and there is not as much a need to have more regions.
        best_n_clusters = max(arr_score, key=arr_score.get) 

        #6b.ii ts. clustering based off good number.
        model = TimeSeriesKMeans(n_clusters=best_n_clusters, metric=metric,
                                 max_iter=50, random_state=10)
        model.fit(X)
        labels = model.labels_
        gdf['ts_labels'][gdf['region']==region] += labels #clusters of region 1 range from 1000 to 1000 + # of clusters in region 1,  clusters of region 2 ....
        print(str(best_n_clusters)+' clusters.') 
#%%Plotting clusters
fig, ax = plt.subplots(figsize=(8,6))
# plot map on axis
countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
countries.plot(color="lightgrey",ax=ax)
# plot points

cmap = cm.get_cmap('gist_rainbow', len(gdf['ts_labels'].unique()) )
color_list = [mcolors.rgb2hex(cmap(i)) for i in range(cmap.N)]
plot = gdf.plot(x="lon", y="lat", kind="scatter", c='ts_labels', colormap=cmap, ax=ax, legend=True)
# add grid
ax.set_xlim([min(yield_lon),max(yield_lon)])
ax.set_ylim([min(yield_lat),max(yield_lat)])
plt.title("distance: " + metric + "; " + "variables: " + ", ".join(cluster_stats))
ax.grid(b=True, alpha=0.5)
#
a= plt.show()


    
#%%#Plot a sampling of time-series by cluster
f, axs = plt.subplots(nrows=5, ncols=1, figsize=(12, 12))
# Make the axes accessible with single indexing
axs = axs.flatten()
# Start a loop over all the variables of interest
for i,region in enumerate(np.arange(2,7)):
    # select the axis where the map will go
    ax = axs[i]
    # Plot the map
    labels= np.arange(region*1000+1+1,region*1000+2+1)
    colors = cm.get_cmap('gist_rainbow', 5)
    colors = [mcolors.rgb2hex(colors(m)) for m in range(colors.N)]
    for j,label in enumerate(labels):
        for k,series in enumerate(gdf['Yield'][gdf['ts_labels']==label]):
            #print(len(gdf['Yield'][gdf['ts_labels']==label]))
            ax.plot(series,color=colors[j])
    ax.legend(['cluster '+str(label) + ': '+str(len(gdf['Yield'][gdf['ts_labels']==label])) + ' series' for label in labels])
        #count=count+1
    # Remove axis clutter
    # Set the axis title to the name of variable being plotted
    ax.set_title("region " + str(region))
# Display the figure
plt.show()

#%%# visualize the probability distribution of bio-clim stats by region or by cluster. Currently it is by region.
cluster_variables=['yield_mean_scaled','GDD_mean_scaled','Pre_mean_scaled']
tidy_db = gdf.set_index("region")
# Keep only variables used for clustering
tidy_db = tidy_db[cluster_variables]
# Stack column names into a column, obtaining
# a "long" version of the dataset
tidy_db = tidy_db.stack()
# Take indices into proper columns
tidy_db = tidy_db.reset_index()
# Rename column names
tidy_db = tidy_db.rename(
    columns={"level_1": "Attribute", 0: "Values"}
)
# Check out result
tidy_db.head()
import seaborn
# Scale fonts to make them more readable
seaborn.set(font_scale=1.5)
seaborn.set_palette(color_list)
# Setup the facets
facets = seaborn.FacetGrid(
    data=tidy_db,
    col="Attribute",
    hue="region",
    sharey=False,
    sharex=False,
    aspect=2,
    col_wrap=3,
)
# Build the plot from `sns.kdeplot`
_ = facets.map(seaborn.kdeplot, "Values", shade=True)
   


