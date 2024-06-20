from pykalman import KalmanFilter
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import contextily as cx
from shapely.geometry import Point, box

def simple_kallman(lats, longs):
    """Applies simple Kallman filter on given measurements, returns tuple with ([lats], [longs])"""
    measurements = list(zip(lats, longs))

    initial_state_mean = [measurements[0][0], 0,
                        measurements[0][1], 0]
    transition_matrix = [[1, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 1],
                        [0, 0, 0, 1]]
    observation_matrix = [[1, 0, 0, 0],
                        [0, 0, 1, 0]]

    kf = KalmanFilter(transition_matrices = transition_matrix,
                    observation_matrices = observation_matrix,
                    initial_state_mean = initial_state_mean,
                    )
    kf = kf.em(measurements, n_iter=5)
    (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)
    lats, _, longs, _ = list(zip(*smoothed_state_means))
    return lats, longs


# function to 3x3 plot
def create_comparison_plot(gdf, area_dicts, feature_A, feature_B, feature_cmap='plasma', difference_cmap='Spectral'):
    """
    Creates a grid of N by 3 plots, with N being the number of areas,
    for each area it displays values of feature_A, values of feature_B,
    and the differences between those values on each respective row.

    Note that this function improperly raises chained assignment flags
    which are therefore temporarily disabled.
    
    args:
    gdf
    returns:
    fig: final figure
    """

    def get_gdf_visualisation_clip(gdf, mask, feature_A, feature_B, radius=5):
        pd.options.mode.chained_assignment = None  # default='warn'
        gdf_clip = gdf.clip(mask)
        gdf_clip['box'] = gdf_clip['geometry'].apply(lambda point: box(*point.buffer(radius).bounds))
        gdf_clip = gdf_clip.set_geometry('box')
        gdf_clip['difference'] = gdf_clip[feature_A] - gdf_clip[feature_B]
        pd.options.mode.chained_assignment = 'warn'  # was None
        return gdf_clip
    
    def ax_default_settings(ax, xmin, ymin, xmax, ymax, crs='EPSG:28992'):
        """
        Update ax for a geo plot to remove axis labels,
        enforce a consistent boundary box, and insert background map
        """
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([]); ax.set_yticks([])
        cx.add_basemap(ax, source=cx.providers.nlmaps.grijs, 
                       attribution=False, crs=crs)
    
    fig, axs = plt.subplots(nrows=3, ncols=len(area_dicts),
                            figsize=(16,13))
    max_dif = min_dif = 0
    
    # creating smaller dfs with info for visualizations
    area_dfs = {area: get_gdf_visualisation_clip(gdf, mask, feature_A, feature_B) 
                for area, mask in area_dicts.items()}
    
    # preparing colour maps and bars for feature plots
    min_feature = min([min(df[feature_A].min(), df[feature_B].min()) 
                       for df in area_dfs.values()])
    max_feature = max([max(df[feature_A].max(), df[feature_B].max()) 
                       for df in area_dfs.values()])
    feature_norm = mpl.colors.Normalize(vmin=round(min_feature), vmax=round(max_feature))
    # preparing colour maps and bars for difference plots
    min_dif = min([df['difference'].min() for df in area_dfs.values()])
    max_dif = min([df['difference'].max() for df in area_dfs.values()])
    difference_norm = mpl.colors.CenteredNorm(vcenter=0,halfrange=max(abs(max_dif),abs(min_dif)))

    #generating sub plots
    for i, (area, _) in enumerate(area_dicts.items()):
        area_gdf = area_dfs[area] 
        xmin, ymin, xmax, ymax = area_gdf.total_bounds
        #plot feature A
        ax = axs[0,i]
        area_gdf.plot(column=feature_A, ax=ax,
                      cmap=feature_cmap, norm=feature_norm)
        ax_default_settings(ax, xmin, ymin, xmax, ymax)
        ax.set_title(area)
        if i == 0: ax.set_ylabel(feature_A)
        #plot feature B
        ax = axs[1,i]
        area_gdf.plot(column=feature_B, ax=ax,
                      cmap=feature_cmap, norm=feature_norm)
        ax_default_settings(ax, xmin, ymin, xmax, ymax)
        if i == 0: ax.set_ylabel(feature_B)
        #plot difference
        ax = axs[2,i]
        area_gdf['difference'] = area_gdf[feature_A] - area_gdf[feature_B]
        area_gdf.plot(column='difference', ax=ax,
                      cmap=difference_cmap, norm=difference_norm)
        ax_default_settings(ax, xmin, ymin, xmax, ymax)
        if i == 0: ax.set_ylabel('Difference')
    

    # inserting colour bars
    fig.colorbar(mpl.cm.ScalarMappable(norm=feature_norm, cmap=feature_cmap), ax=(axs[0,0], axs[0,1], axs[0,2]))
    fig.colorbar(mpl.cm.ScalarMappable(norm=feature_norm, cmap=feature_cmap), ax=(axs[1,0], axs[1,1], axs[1,2]))
    fig.colorbar(mpl.cm.ScalarMappable(norm=difference_norm, cmap=difference_cmap), ax=(axs[2,0], axs[2,1], axs[2,2]))

    return fig