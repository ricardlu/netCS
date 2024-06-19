# -*- coding: utf-8 -*-
"""
@author: lricard
"""

import numpy as np
import matplotlib.pyplot as plt

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


if __name__ == '__main__':   
    # %matplotlib qt
    
    names_CMIP6 = np.array(['NorESM2-MM (3)', 'NorESM2-LM (3)', 'MPI-HR (10)', 'MPI-LR (30)', 'EC-Earth3 (14)', 
                            'EC-Earth-Veg-LR (3)', 'MIROC6 (50)', 'GFDL-ESM4 (2)', 'GISS-E2-1-G (12)',
                            'HadGEM3 (5)', 'CNRM-CM6-1 (26)', 'CanESM5 (24)', 'INM-CM5 (10)', 'UKESM1-0-LL (17)', 
                            'IPSL (32)', 'NESM3 (5)', 'MRI-ESM2-0 (10)',  'BCC-ESM1 (3)', 'BCC-CSM2-MR (3)',
                            'EC-Earth_AerChem (2)', 'SAMO UNICON (1)', 'ACCESS-ESM1-5 (15)', 'FGOALS-f3-L (3)',
                            'ESM-1-0 (5)', 'CAMS-CSM1-0 (2)', 'CESM2 (11)', 'CESM2 WACCM (3)'])
    
    N_models = len(names_CMIP6)
    rep = 'CMIP6_metrics/'
    
    ''' Load regions'''
    N_models = len(names_CMIP6)
    d_maps1 = np.load ('Data/regions_reference_HadISST_sst_1975-2014.npy')
    N_regions = len(d_maps1)
    size_regions = np.array([np.nansum(d_maps1[i,]) for i in range (len(d_maps1))])

    d_ids1 = np.arange(1, N_regions +1)
    domain_map1 = build_domain_map (d_maps1)
    domain_map1_plot = maps_to_plot (domain_map1)
    
    '''Distance ACE '''   
    # Array size number of models
    intra_mean_D_ACE_HadISST = np.load(rep + 'intra_model_mean_9regions_D_ACE_ref_HadISST_1975-2014.npy')
    intra_mean_D_ACE_COBEv2 = np.load(rep + 'intra_model_mean_9regions_D_ACE_ref_COBEv2_1975-2014.npy')
    intra_mean_D_ACE_mix = np.load(rep + 'intra_model_mean_9regions_D_ACE_ref_MIX_1975-2014.npy')  
    
    intra_std_D_ACE_HadISST = np.load(rep + 'intra_model_std_9regions_D_ACE_ref_HadISST_1975-2014.npy')
    intra_std_D_ACE_COBEv2 = np.load(rep + 'intra_model_std_9regions_D_ACE_ref_COBEv2_1975-2014.npy')
    intra_std_D_ACE_mix = np.load(rep + 'intra_model_std_9regions_D_ACE_ref_MIX_1975-2014.npy')   
    
    # Array size number of models filled with array number of runs
    import_D_ACE_HadISST = np.load(rep + '9regions_D_ACE_ref_HadISST_1975-2014.npy', allow_pickle =True)
    import_D_ACE_COBEv2 = np.load(rep + '9regions_D_ACE_ref_COBEv2_1975-2014.npy', allow_pickle =True)
    import_D_ACE_mix = np.load(rep + '9regions_D_ACE_ref_MIX_1975-2014.npy', allow_pickle =True)
  
  
    
    '''WWD'''
    # Array size number of models
    intra_mean_WWD = np.load(rep + 'intra_mean_WWD_ref_HadISST_1975-2014.npy')
    intra_mean_WWD2 = np.load(rep + 'intra_mean_WWD_ref_COBEv2_1975-2014.npy')
    intra_mean_WWD_mix = np.load(rep + 'intra_mean_WWD_ref_MIX_1975-2014.npy')
    
    intra_std_WWD = np.load(rep + 'intra_std_WWD_ref_HadISST_1975-2014.npy')
    intra_std_WWD2 = np.load(rep + 'intra_std_WWD_ref_COBEv2_1975-2014.npy')
    intra_std_WWD_mix = np.load(rep + 'intra_std_WWD_ref_MIX_1975-2014.npy')
    
    # Array size number of models filled with array number of runs
    import_WWD_HadISST = np.load (rep + 'WWD_model_ref_HadISST_1975-2014.npy', allow_pickle =True)
    import_WWD_COBEv2 = np.load (rep + 'WWD_model_ref_COBEv2_1975-2014.npy', allow_pickle =True)
    import_WWD_mix = np.load (rep + 'WWD_model_ref_MIX_1975-2014.npy', allow_pickle =True)
    
    # Array size number of models filled with array number of runs*number of regions
    mat_contrib_WWD_HadISST = np.load (rep + 'Mat_contrib_WWD_model_ref_HadISST_1975-2014.npy', allow_pickle =True)
    mat_contrib_WWD_COBEv2 = np.load (rep + 'Mat_contrib_WWD_model_ref_COBEv2_1975-2014.npy', allow_pickle =True)
    mat_contrib_WWD_mix = (mat_contrib_WWD_HadISST + mat_contrib_WWD_COBEv2)/2
    
    # Contribution ENSO
    ratio_ENSO = []
    for i in range (N_models):
        ratio_ENSO_runs = mat_contrib_WWD_mix[i][:,0]/np.nansum(mat_contrib_WWD_mix[i][:,1:], axis = 1)
        ratio_ENSO.append (ratio_ENSO_runs)
    ratio_ENSO_models = np.array([np.nanmean (ratio_ENSO[i]) for i in range (N_models)])
   

    # Inter-mean and std WWD
    inter_mean_WWD =  np.nanmean (intra_mean_WWD)
    inter_std_WWD =  np.nanstd (intra_mean_WWD)
    
    inter_mean_WWD2 =  np.nanmean (intra_mean_WWD2)
    inter_std_WWD2 =  np.nanstd (intra_mean_WWD2)
    
    inter_mean_WWD_mix =  np.nanmean (intra_mean_WWD_mix)
    inter_std_WWD_mix = np.nanstd (intra_mean_WWD_mix)
    

    # Internal variability and inter-model variability
    print (len(np.where (intra_std_WWD > inter_std_WWD)[0]))
    print (len(np.where (intra_std_WWD2 > inter_std_WWD2)[0]))
    print (len(np.where (intra_std_WWD_mix > inter_std_WWD_mix)[0]))

    # Obs uncertainty    
    print ('Correlation WWD (HadISST, COBEv2) =', np.corrcoef (intra_mean_WWD, intra_mean_WWD2)[0,1].round(3))
    

    ''' Internal variability & Observational Uncertainty '''
    ### For each model : ensemble mean and std
    x1 = np.arange(1,N_models+1)*5 -1
    x2 = x1 +1
    
    name_diag  = '$WWD$'
    # name_diag = '$D_{ACE}$'
    
    diag =  intra_mean_WWD
    diag2 = intra_mean_WWD2
    diag3 = intra_mean_WWD_mix
    std_diag = intra_std_WWD
    std_diag2 = intra_std_WWD2
    std_diag3 =  intra_std_WWD_mix
    
    # diag =  intra_mean_D_ACE_HadISST
    # diag2 = intra_mean_D_ACE_COBEv2
    # diag3 = intra_mean_D_ACE_mix
    # std_diag = intra_std_D_ACE_HadISST
    # std_diag2 = intra_std_D_ACE_COBEv2
    # std_diag3 =  intra_std_D_ACE_mix
   
    mu1, sigma1 = diag.mean().round(2), diag.std().round(2)
    mu2, sigma2 = diag2.mean().round(2), diag2.std().round(2)
    mu3, sigma3 = diag3.mean().round(2), diag3.std().round(2)
    
    plt.figure (figsize = (10,6), dpi = 150)
    ax = plt.gca()
    ax.axhline (mu1, linestyle = '--', color = 'b')
    ax.axhline (mu2, linestyle = '--', color = 'r')
    plt.errorbar(x=x1, y=diag, yerr=std_diag, color = 'darkblue', fmt = '_', capsize=4)
    plt.errorbar(x=x2, y=diag2, yerr=std_diag2, color = 'r', fmt = '_', capsize=4)
    plt.plot (x1, diag, 'o', color = 'darkblue', label = 'ref HadISST (\u03BC = %s +/- \u03C3 = %s)' %(mu1, sigma1))
    plt.plot (x2, diag2, 'o', color = 'r', label = 'ref COBEv2 (\u03BC = %s +/- \u03C3 = %s)' %(mu2, sigma2))
    plt.legend()
    plt.xlabel ('CMIP6 models', size = 16)
    plt.ylabel ('Intra-model mean and std %s' %name_diag, size = 16)
    plt.xticks (x1+0.25, labels= names_CMIP6, fontsize = 11, rotation = 90)
    plt.yticks (fontsize = 13)
    plt.grid(alpha = .3)
    plt.tight_layout()
    # plt.savefig ('Figures/Internal_variability_%s' %name_diag, dpi = 300)
    
    plt.figure (figsize = (10,6), dpi = 150)
    ax = plt.gca()
    ax.axhline (mu3 + sigma3, linestyle = '--', color = 'grey')
    ax.axhline (mu3 - sigma3, linestyle = '--', color = 'grey')
    ax.axhline (mu3, linestyle = '-', color = 'darkgrey', label = '\u03BC = %s +/- \u03C3 = %s' %(mu3, sigma3))
    plt.errorbar(x=x1, y=diag3, yerr=std_diag3, color = 'k', fmt = '_', capsize=4)
    plt.plot (x1, diag3, 'o', color = 'k')
    plt.legend()
    plt.xlabel ('CMIP6 models', size = 16)
    plt.ylabel ('Intra-model mean and std %s' %name_diag, size = 16)
    plt.xticks (x1+0.25, labels= names_CMIP6, fontsize = 11, rotation = 90)
    plt.yticks (fontsize = 13)
    plt.grid(alpha = .3)
    plt.tight_layout()

    # plt.savefig ('Figures/Internal_variability_mix_%s' %name_diag, dpi = 300)
    
    
    '''Contribution region'''
    
    plt.figure (figsize = (10,6), dpi = 150)
    for k in range (27):
        n_runs, _ = mat_contrib_WWD_HadISST[k].shape
        plt.plot (np.ones((n_runs))*x1[k], ratio_ENSO[k], '.', color = 'dodgerblue')
    plt.plot (x1, ratio_ENSO_models, 'o', color = 'b')
    plt.legend()
    plt.xlabel ('CMIP6 models', size = 16)
    plt.ylabel ('Value of Weighted Wasserstein Distance in ENSO region %s' %name_diag, size = 16)
    plt.xticks (x1+0.25, labels= names_CMIP6, fontsize = 11, rotation = 90)
    plt.yticks (fontsize = 13)
    plt.grid(alpha = .3)
    plt.tight_layout()

    
    
    
    
