# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 09:43:39 2023

@author: danie

#
Regionalize by the climatology of the agro-clim variables outputted by MakeBioClimVars.py, then cluster the time series within these regions by euclidean k-means.
#
"""

import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
pd.options.mode.chained_assignment = None  # default='warn'
#%%0. Load all input data.

#0a. Write down the file paths.
GDD_path = "./Bioclim_vars/gepic_hadgem2-es_ewembi_historical_2005soc_co2_GDD_growing_season-soy-noirr.nc4"
Pre_path = './Bioclim_vars/pr_cumul_growing_season_HadGEM2-ES_historical_r1i1p1_EWEMBI_europe_18610101-20051231.nc4'
yield_path="./Bioclim_vars\gepic_hadgem2-es_ewembi_historical_2005soc_co2_yield-soy-noirr_lat36.25to70.25lon-11.25to40.25_annual_1861_2005.nc4";
precip_path = "./Bioclim_vars/pr_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_europe_18610101-20051231.nc4"
tasmax_path = "./Bioclim_vars/tasmax_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_europe_18610101-20051231.nc4"
tasmin_path = "./Bioclim_vars/tasmin_day_HadGEM2-ES_historical_r1i1p1_EWEMBI_europe_18610101-20051231.nc4"
plant_day_path = "./Bioclim_vars/gepic_hadgem2-es_ewembi_historical_2005soc_co2_plantday-soy-noirr_lat36.25to70.25lon-11.25to40.25_annual_1861_2005.nc4"
mature_day_path = "./Bioclim_vars/gepic_hadgem2-es_ewembi_historical_2005soc_co2_matyday-soy-noirr_lat36.25to70.25lon-11.25to40.25_annual_1861_2005.nc4"
#0b. Load in the files as netCDF4. We will join them in a pandas dataframe shortly.
tasmax = netCDF4.Dataset(tasmax_path)['tasmax']
tasmin = netCDF4.Dataset(tasmin_path)['tasmin']
precip = netCDF4.Dataset(precip_path)['pr']
plant_day = netCDF4.Dataset(plant_day_path)['plantday-soy-noirr']
mature_day = netCDF4.Dataset(mature_day_path)['matyday-soy-noirr']
GDD = netCDF4.Dataset(GDD_path)['GDD']
Pre = netCDF4.Dataset(Pre_path)['pre']
soy_yield = netCDF4.Dataset(yield_path)['yield-soy-noirr']

yield_lat = netCDF4.Dataset(yield_path)['lat']
yield_lon =netCDF4.Dataset(yield_path)['lon']
#%%1. Making a central dataframe, to fill with relevant quantities for each pixel [1:4248].

#1a. Compute long-term averages of the agro-clim variables. 
#Filling the central dataframe with these quantities.
Agro_Clim_variables=['GDD_mean','Pre_mean','GDD_var','Pre_var','yield_mean','yield_std','yield_max','yield_min']
count = 0;
a = [None]*(69*104)
b={}
for i in np.arange(0,69):
    for j in np.arange(0,104):
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
print(count) #monitor loop progress
Agro_Clim_stats=pd.DataFrame(a,columns=Agro_Clim_variables)

#%%1b. Filling the central dataframe with yield time-series + a variable which gives a 2-D enumeration of the lat/lon pairings. 
ts_variables=['Yield','lat-lon'] 
count = 0;
a = [None]*(69*104)
for i in np.arange(0,len(yield_lat)):
    for j in np.arange(0,len(yield_lon)):
        soy_slice = soy_yield[:,i,j]
        latlon = (i,j)
        if (soy_slice.mask.sum() == 0) & (GDD[:,i,j].mask.sum()==0):  
                a[count] = (soy_slice.data,
                            latlon,
                            count)
                b.update({count:(i,j)})
                count = count + 1
        else: 
            a[count] = (np.nan,latlon,count)
            count = count + 1
print(count)
Temporal_Variables = pd.DataFrame(a,columns=ts_variables+['count'])
#%%2. Converting the central dataframe into a pandas format.
xm,ym = np.meshgrid(yield_lon[:],yield_lat[:]);
data = {'lon':xm.flatten(),'lat':ym.flatten(),}
coordinates = pd.DataFrame.from_dict(data);
df = pd.concat([coordinates,Temporal_Variables,Agro_Clim_stats],axis=1)

#%%. Plot: Spatially plot some chosen mean statistics 
cluster_variables=['GDD_mean','Pre_mean','yield_mean']
f, axs = plt.subplots(nrows=len(cluster_variables), ncols=1, figsize=(12, 12))
# Make the axes accessible with single indexing
axs = axs.flatten()
titles = ['Growing Degree Days','Precipitation','Yield']
Units = ['Kelvin','$kg$ x $m^{-2}$','$Tons$ x $Hectare^{-1}$']
# Start a loop over all the variables of interest
for i, col in enumerate(cluster_variables):
    # select the axis where the map will go
    ax = axs[i]
    # Plot the map
    df.plot(x="lon", y="lat", kind="scatter", c=col, colormap="YlOrRd", ax=ax)
    # Remove axis clutter
    ax.set_axis_off()
    # Set the axis title to the name of variable being plotted
    ax.set_title(titles[i],fontsize=22)
    ax.get_figure().axes[-1].set_ylabel(Units[i], size=20)
# Display the figure
plt.show()
#%%3. Transform again the pandas dataframe into a geopandas dataframe.
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
#%%4. Define a distance matrix which will mark contiguity of pixels. For later use in regionalization.
import numpy
w = Queen.from_dataframe(gdf_points)
#%%5. Scale the agro-clim statistics using a robust scale. Robust scaling minimizes the effects of outliers.
import sklearn
from sklearn.preprocessing import robust_scale, scale, minmax_scale
bioclim_stats = ['GDD_mean','Pre_mean','yield_mean']
scaled_stats = robust_scale(gdf[bioclim_stats]) #scales each column

#Save the scaled statistics in the dataframe too.
gdf['GDD_mean_scaled']=robust_scale(gdf['GDD_mean'])
gdf['Pre_mean_scaled']=robust_scale(gdf['Pre_mean'])
gdf['yield_mean_scaled']=robust_scale(gdf['yield_mean'])
#%%6. Compute regions of the data based on clustering by the scaled agro-clim statistics
#6a. Identify optimal number of regions using sillouhette scoring
from scipy.cluster.hierarchy import dendrogram, linkage

#use a euclidean metric.
metric = 'euclidean'
linkage_method = 'ward'
X = scaled_stats
Z = linkage(X, metric=metric, method=linkage_method)

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

#Now, choose the number of regions. A heuristic is to take the # before a big drop off in score.
best_n_regions = max(arr_score, key=arr_score.get)
print(best_n_regions)
best_n_regions=int(input('Set n regions.'))
#%%6b. Compute the optimal regionalization, and label the dataframe with these regional labels.
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
#%%Plot: Plotting regions
fig, ax = plt.subplots(figsize=(8,6))
# plot map on axis
countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
countries.plot(color="lightgrey",ax=ax)
# plot points
cmap = cm.get_cmap('gist_rainbow', best_n_regions)
color_list = [mcolors.rgb2hex(cmap(i)) for i in range(cmap.N)]
plot = gdf.plot(x="lon", y="lat", kind="scatter", c='region', colormap=cmap, ax=ax, legend=True)
colorbar_ax = ax.get_figure().axes[-1] #to get the last axis of the figure, it's the colorbar axes
colorbar_ax.set_ylabel("Agroclimatic Region", size=20)
colorbar_ax.tick_params(labelsize=15)
tick_locs = (np.arange(best_n_regions) + 0.5)*(best_n_regions-1)/best_n_regions
colorbar_ax.set_yticks(tick_locs)
colorbar_ax.set_yticklabels(np.arange(best_n_regions))
# add grid
ax.set_xlim([min(yield_lon),max(yield_lon)])
ax.set_ylim([min(yield_lat),max(yield_lat)])
plt.xticks([])
ax.get_yaxis().set_visible(False)
ax.set_xlabel('Northern Europe',fontsize=20,labelpad=20)
a= plt.show()


#%%Plot: Visualizing the regional data distributions
x,y,z=gdf['GDD_mean'],gdf['Pre_mean'],gdf['yield_mean']
import plotly.graph_objects as go
import plotly.express as px
import plotly as py

gdf['region_str']=gdf['region'].astype(str)
fig = px.scatter_3d(gdf, x="GDD_mean_scaled", y="Pre_mean_scaled", z="yield_mean_scaled",color="region_str",color_discrete_sequence=color_list,
                    category_orders={"region_str": list(np.arange(0,9).astype(str))})
fig.update_traces(marker=dict(size=2))
fig.show()
#py.offline.plot(fig, filename = './1a_Regions_Yield_3DScatter.html')
#%%7a. Prepare time-series data for clustering by scaling.
cluster_ts = ['Yield']
gdf_scaled_variables = np.ndarray((len(gdf),145,len(cluster_ts)))
for i in np.arange(0,len(gdf),1):
    for j in np.arange(len(cluster_ts)):
        print(i)
        gdf_scaled_variables[i,:,j]= scale(gdf[cluster_ts[j]].iloc[i][:])
          
#%%7b. Compute time-series clusters by k-means. An important parameter here is max_k, which is the highest cluster number which will enter the competition for optimality.
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
        #Plot the range of cluster scores for each region.
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.bar(list(arr_score.keys()),list(arr_score.values()))
        ax.set_xlabel('# clusters')
        ax.set_ylabel('score')
        plt.title('Cluster scores for region '+ str(region))
        plt.show()
        ###Here, simply choose the number of clusters with the highest score. We just want to cluster very similar time series, and there is not as much a need to have more regions.
        best_n_clusters = max(arr_score, key=arr_score.get) 

        #Compute the final clustering based off the prefered cluster numbers found above.
        model = TimeSeriesKMeans(n_clusters=best_n_clusters, metric=metric,
                                 max_iter=50, random_state=10)
        model.fit(X)
        labels = model.labels_
        gdf['ts_labels'][gdf['region']==region] += labels #clusters of region 1 range from 1000 to 1000 + # of clusters in region 1,  clusters of region 2 ....
        print(str(best_n_clusters)+' clusters.') 
#%%Plot: Plotting clusters
fig, ax = plt.subplots(figsize=(8,6))
# plot map on axis
countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
countries.plot(color="lightgrey",ax=ax)
# plot points

cmap = cm.get_cmap('gist_rainbow', len(gdf['ts_labels'].unique()) )
color_list = [mcolors.rgb2hex(cmap(i)) for i in range(cmap.N)]
plot = gdf.plot(x="lon", y="lat", kind="scatter", c='ts_labels', colormap=cmap, ax=ax, legend=True)
# add grid
colorbar_ax = ax.get_figure().axes[-1] #to get the last axis of the figure, it's the colorbar axes
colorbar_ax.set_ylabel("Agroclimatic Region", size=20)
colorbar_ax.tick_params(labelsize=15)
tick_locs = (np.arange(best_n_regions) + 0.5)*(best_n_regions-1)/best_n_regions
colorbar_ax.set_yticks(tick_locs)
colorbar_ax.set_yticklabels(np.arange(best_n_regions))
# add grid
ax.set_xlim([min(yield_lon),max(yield_lon)])
ax.set_ylim([min(yield_lat),max(yield_lat)])
plt.xticks([])
ax.get_yaxis().set_visible(False)
ax.set_xlabel('Europe',fontsize=20,labelpad=20)

ax.set_xlim([min(yield_lon),max(yield_lon)])
ax.set_ylim([min(yield_lat),max(yield_lat)])

plt.title("distance: " + metric + "; " + "variables: " + ", ".join(cluster_ts))
ax.grid(b=True, alpha=0.5)
#
a= plt.show()
    
#%%Plot: Plot a sampling of time-series by cluster
subplot_labels = [['a)'],['b)'],['c)'],['d)'],['e)']];
f, axs = plt.subplots(nrows=5,ncols=1,figsize=(12, 12))
# Make the axes accessible with single indexing
axs = axs.flatten()
plt.style.use('default')

# Start a loop over all the variables of interest
selected_regions = np.arange(1,6)
for i,region in enumerate(selected_regions):
    # select the axis where the map will go
    ax = axs[i]
    # Plot the map
    labels= np.arange(region*1000+4,region*1000+5)
    cmap = cm.get_cmap('gist_rainbow', best_n_regions)
    color_list = [mcolors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    for j,label in enumerate(labels):
        for k,series in enumerate(gdf['Yield'][gdf['ts_labels']==label]):
            #print(len(gdf['Yield'][gdf['ts_labels']==label]))
            ax.plot(series,color='black')
    ax.legend(['Cluster '+str(label) + ': '+str(len(gdf['Yield'][gdf['ts_labels']==label])) + ' series' for label in labels],loc='lower right',fontsize=20,frameon=True)
        #count=count+1
    # Remove axis clutter
    # Set the axis title to the name of variable being plotted
    if i != len(selected_regions)-1:
        ax.get_xaxis().set_visible(False)
    ax.set_xlabel('Years',fontsize=25)
    ax.set_xlim(-1,144)
    ax.text(0.01, 0.8, 'Region '+str(region),
    transform = ax.transAxes,fontsize=25, color='black', bbox=dict(facecolor=color_list[region], alpha=0.5))
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_ylim((ax.get_ylim()[0],ax.get_ylim()[1]*1.07))
# Display the figure
#f.supylabel('Yield [T/H]')
axs[2].set_ylabel('Yield [$Tons$ x $Hectare^{-1}$]',fontsize=25)
f.tight_layout()

plt.show()

#%%#Plot: visualize the probability distribution of bio-clim stats by region or by cluster. Currently it is by region.
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
# Setup the facets
facets = seaborn.FacetGrid(
    data=tidy_db,
    col="Attribute",
    hue="region",
    sharey=False,
    sharex=False,
    aspect=2,
    col_wrap=1,
)
# Build the plot from `sns.kdeplot`
_ = facets.map(seaborn.kdeplot, "Values", shade=True)
   
