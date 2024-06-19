# -*- coding: utf-8 -*-
"""
@author: lricard
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset
import glob
from sklearn.metrics.cluster import normalized_mutual_info_score 
import matplotlib as mpl
import cmasher as cmr
import scipy
from scipy.stats import spearmanr, kurtosis, skew, moment
import seaborn as sns

#Compute signals
def cumulative_anomaly(data,domain):
    return np.nansum(data*domain,axis=(1,2))

def average_anomaly(data,domain):
    return np.nanmean(data*domain,axis=(1,2))

def maps_to_plot (d_maps):
    d_maps_to_plot = d_maps.copy() * np.nan
    d_maps_to_plot[d_maps != 0] = d_maps[d_maps != 0]
    return d_maps_to_plot

def build_domain_map (d_maps):
    #Build domain map[x,y] from d_maps [d,x,y]
    #Put 0 if grid cells belongs to any domains and i if belongs to i-th domain
    d,x,y = d_maps.shape
    domain_map_output = np.zeros((x,y))
    #Read in reverse direction because index 0 is stronger than index 31
    for d in range(len(d_maps)) [::-1]:     
        domain_map_output[d_maps[d] == 1] = d+1
    return domain_map_output

def normalize (signals):
    stds = np.std(signals, axis = 1)
    signals_norm = []
    for i in range (len(signals)) :
        signals_norm.append(signals[i,:]/stds[i])
    signals_norm = np.array(signals_norm)
    return signals_norm

def compute_signals (inname, d_maps):
    file= Dataset(inname, 'r')
    try :
        sst = file['sst'][:]
    except :
        sst = file ['tos'][:]
    sst_filled = np.ma.filled(sst.astype(float), np.nan)
    # print(np.nanmax(sst_filled).round(3))
    # print(np.nanmin(sst_filled).round(3))
    # print('\n')
    weighted_domains = d_maps*lat_weights 
    signals = []
    for i in range(len(weighted_domains)):
        # signals.append(average_anomaly(sst_filled,weighted_domains[i]))
        signals.append(cumulative_anomaly(sst_filled, weighted_domains[i]))
    signals = np.array(signals)
    norm_signals = normalize (signals)
    return signals, norm_signals


if __name__ == '__main__':   
    %matplotlib qt
    
    rep = 'CMIP6_metrics/'
    
    # Longitude-Latitude
    lat = np.arange(-59.,60.,2)
    lon = np.arange (0.,360.,2)
    LON, LAT = np.meshgrid(lon, lat)
    lat_rad = np.radians(lat)
    lat_weights = np.cos(lat_rad).reshape(len(lat_rad),1)
    
    # Load SST regions inferred with delta-MAPS in HadISST dataset
    d_maps1 = np.load ('Data/regions_reference_HadISST_sst_1975-2014.npy')
    N_regions = len(d_maps1)
    print ('Number SST regions = %s'%N_regions)
    d_ids1 = np.arange(1, N_regions +1)
    domain_map1 = build_domain_map (d_maps1) #filled with 0 and 1 
    domain_map1_plot = maps_to_plot (domain_map1) #filled with NaN and 1
    
    strengths = np.load ('Data/strength_list_HadISST_sst_1975-2014.npy')
    weights_strengths = strengths[:,1]/np.sum(strengths[:,1])  
    weights_equal = np.ones((N_regions))*1/N_regions
    
    # Cumulative sum of normalized strength "weight strength" regions
    plt.figure(figsize = (5,4), dpi = 150)
    plt.plot (np.arange(1,N_regions+1), np.cumsum(weights_strengths), marker = '.', color= 'b')
    plt.ylabel ('Cumulative sum of normalized strengths', size = 12)
    plt.xlabel ('SST regions')
    plt.grid()
    plt.tight_layout()
    
    # Calculate signals (i.e. SST timeseries in regions) - Shape number of regions*number of months 
    filename_HadISST = 'Data/processed_HadISST_sst_2x2_197501-201412.nc'
    signals_HadISST, signals_norm_HadISST = compute_signals (filename_HadISST, d_maps1)
    print (signals_HadISST.shape)
    
    filename_COBEv2 = 'Data/processed_COBEv2_sst_2x2_197501-201412.nc'
    signals_COBEv2, signals_norm_COBEv2 = compute_signals (filename_COBEv2, d_maps1)
    print (signals_COBEv2.shape)
    
    vect_time = pd.date_range(start='1/1/1975', end='1/1/2015', freq = 'M')
    print (len(vect_time))

    # We can calculate the WD (COBEv2, HadISST) 
    vect_WWD_obs = []
    for i in range (len(signals_HadISST)):
        WD_value = scipy.stats.wasserstein_distance (signals_HadISST[i,], signals_COBEv2[i,])
        WWD_value = WD_value*weights_strengths[i]     
        vect_WWD_obs.append (WWD_value)
    vect_WWD_obs = np.array(vect_WWD_obs)
    
    ## Retrieve mean, std, skewness and kurtosis 
    # std_domains = np.std(signals_HadISST, axis = 1)
    # skewness_domains = scipy.stats.skew(signals_HadISST, axis = 1)
    # kurtosis_domains = scipy.stats.kurtosis(signals_HadISST, axis = 1)
    
    
    list_model_names = ['NorESM2-MM', 'NorESM2-LM', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'EC-Earth3',
                        'EC-Earth3-AerChem', 'EC-Earth3-Veg-LR', 'MIROC6', 'GFDL-ESM4', 'GISS-E2-1-G',
                        'HadGEM3-GC31-LL', 'CNRM-CM6-1', 'CanESM5', 'INM-CM5', 'UKESM1-0-LL',
                        'IPSL-CM6A-LR', 'NESM3', 'MRI-ESM2-0', 'BCC-ESM1', 'BCC-CSM2-MR', 'SAM0-UNICON',
                        'ACCESS-ESM1-5', 'E3SM-1-0', 'FGOALS-f3-L', 'CAMS-CSM1-0', 'CESM2', 'CESM2-WACCM']
    
    N_models = len (list_model_names)

    # For each member of each CMIP6 model, calculate and save WWD
    d_ids3 = np.copy(d_ids1)
    d_maps3 = np.copy(d_maps1)
   
    WWD_of_models_ref_HadISST = []   
    WWD_of_models_ref_COBEv2 = [] 
    mat_contrib_in_models_ref_HadISST = []
    mat_contrib_in_models_ref_COBEv2 = []

    for i, model_name in enumerate(list_model_names):
    
        # Path of historical outputs
        nc_files = glob.glob ('Data/Data-%s/*.nc' %(model_name))
        number_runs = len(nc_files)
        print ('Model %s - number member id = %s' %(model_name, number_runs))
        
        
        vect_WWD_in_regions_all_runs_ref_HadISST = np.ones((N_regions))*np.nan
        vect_WWD_in_regions_all_runs_ref_COBEv2 = np.ones((N_regions))*np.nan
        
        vect_WWD_ref_HadISST = []
        vect_WWD_ref_COBEv2 = []

        for i, path_run in enumerate (nc_files): 
            run_name = 'run_%s' %i
            signals_simu, signals_norm_simu = compute_signals (path_run, d_maps3)
            
            sum_WWD_ref_HadISST, sum_WWD_ref_COBEv2 = 0, 0
            vect_WWD_in_regions_ref_HadISST = []
            vect_WWD_in_regions_ref_COBEv2 = []
            
            for k in range (len(signals_HadISST)):
                WD_value_ref_HadISST = scipy.stats.wasserstein_distance (signals_HadISST[k,], signals_simu[k,])
                WD_value_ref_COBEv2 =  scipy.stats.wasserstein_distance (signals_COBEv2[k,], signals_simu[k,])
                WWD_value_ref_HadISST = WD_value_ref_HadISST *weights_strengths[k]    
                WWD_value_ref_COBEv2 = WD_value_ref_COBEv2 *weights_strengths[k]    
                sum_WWD_ref_HadISST += WWD_value_ref_HadISST
                sum_WWD_ref_COBEv2 += WWD_value_ref_COBEv2
                vect_WWD_in_regions_ref_HadISST.append (WWD_value_ref_HadISST)
                vect_WWD_in_regions_ref_COBEv2.append (WWD_value_ref_COBEv2)
             
            # Sum of WWD of model i, run k 
            vect_WWD_ref_HadISST.append (sum_WWD_ref_HadISST)
            vect_WWD_ref_COBEv2.append (sum_WWD_ref_COBEv2)
            
            # all WWD values in regions of model i, run k
            vect_WWD_in_regions_all_runs_ref_HadISST =  np.vstack((vect_WWD_in_regions_all_runs_ref_HadISST, vect_WWD_in_regions_ref_HadISST))
            vect_WWD_in_regions_all_runs_ref_COBEv2 =  np.vstack((vect_WWD_in_regions_all_runs_ref_COBEv2, vect_WWD_in_regions_ref_COBEv2))
        
       
        # List of CMIP6 models with array containing WWD values of the runs
        WWD_of_models_ref_HadISST.append (np.array(vect_WWD_ref_HadISST))
        WWD_of_models_ref_COBEv2.append (np.array(vect_WWD_ref_COBEv2))
            
        # Remove the first artefact and get list of CMIP6 models with array n_runs*n_regions
        vect_WWD_in_regions_all_runs_ref_HadISST = vect_WWD_in_regions_all_runs_ref_HadISST[1:,:]
        vect_WWD_in_regions_all_runs_ref_COBEv2 = vect_WWD_in_regions_all_runs_ref_COBEv2[1:,:]
        
        mat_contrib_in_models_ref_HadISST.append (np.array(vect_WWD_in_regions_all_runs_ref_HadISST))
        mat_contrib_in_models_ref_COBEv2.append (np.array(vect_WWD_in_regions_all_runs_ref_COBEv2))
        
    # With respect to both HadISST and COBEv2
    WWD_of_models_ref_MIX = []
    for i in range (len(WWD_of_models_ref_HadISST)):
        WWD_of_models_ref_MIX.append (np.array ((WWD_of_models_ref_HadISST[i]+ WWD_of_models_ref_COBEv2[i])/2))
  
    # We can calculate the intra-model mean and std
    mean_WWD_ref_HadISST = np.array([np.nanmean (WWD_of_models_ref_HadISST[i]) for i  in range(len(WWD_of_models_ref_HadISST))]) 
    mean_WWD_ref_COBEv2 = np.array([np.nanmean (WWD_of_models_ref_COBEv2[i]) for i  in range(len(WWD_of_models_ref_COBEv2))])
    mean_WWD_ref_MIX = np.array([np.nanmean (WWD_of_models_ref_MIX[i]) for i  in range(len(WWD_of_models_ref_MIX))])
     
    std_WWD_ref_HadISST = np.array([np.nanstd (WWD_of_models_ref_HadISST[i]) for i  in range(len(WWD_of_models_ref_HadISST))]) 
    std_WWD_ref_COBEv2 = np.array([np.nanstd (WWD_of_models_ref_COBEv2[i]) for i  in range(len(WWD_of_models_ref_COBEv2))])
    std_WWD_ref_MIX = np.array([np.nanstd (WWD_of_models_ref_MIX[i]) for i  in range(len(WWD_of_models_ref_MIX))])
  
    # Save 
    rep = 'CMIP6_metrics/'
    np.save (rep + 'Mat_contrib_WWD_model_ref_HadISST_1975-2014.npy', mat_contrib_in_models_ref_HadISST)
    np.save (rep + 'Mat_contrib_WWD_model_ref_COBEv2_1975-2014.npy', mat_contrib_in_models_ref_COBEv2)
    
    
    np.save (rep + 'WWD_model_ref_HadISST_1975-2014.npy', WWD_of_models_ref_HadISST)
    np.save (rep + 'WWD_model_ref_COBEv2_1975-2014.npy', WWD_of_models_ref_COBEv2)
    np.save (rep + 'WWD_model_ref_MIX_1975-2014.npy', WWD_of_models_ref_MIX)

    np.save (rep + 'intra_mean_WWD_ref_HadISST_1975-2014.npy', mean_WWD_ref_HadISST)
    np.save (rep + 'intra_std_WWD_ref_HadISST_1975-2014.npy', std_WWD_ref_HadISST)
    np.save (rep + 'intra_mean_WWD_ref_COBEv2_1975-2014.npy', mean_WWD_ref_COBEv2)
    np.save (rep + 'intra_std_WWD_ref_COBEv2_1975-2014.npy', std_WWD_ref_COBEv2)
    np.save (rep + 'intra_mean_WWD_ref_MIX_1975-2014.npy', mean_WWD_ref_MIX)
    np.save (rep + 'intra_std_WWD_ref_MIX_1975-2014.npy', std_WWD_ref_MIX)
    
   
 


       
            

        

        
