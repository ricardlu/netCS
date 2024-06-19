# -*- coding: utf-8 -*-
"""
@author: lricard

We load CMIP6 outputs and preprocess them with CDO
"""

import numpy as np
import xarray as xr
import os
import glob
import intake
import shutil


def execute_cdo_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()
    if stderr:
        print(f"Error occurred: {stderr.decode()}")
    else:
        print(f"Output: {stdout.decode()}")

if __name__ == '__main__': 

    cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
    col = intake.open_esm_datastore(cat_url)
    
    ## To specify
    CMIP6_model_id = 'NorESM2-MM'
    var_id = 'tos'
    member_id = 'r1i1p1f1'
    year1, year2 = 1975, 2014
    name_save = 'historical_%s_%s' %(year1, year2)
    
    ## Search CMIP6 data 
    cat = col.search(source_id = CMIP6_model_id, 
                  experiment_id= 'historical', 
                  table_id='Omon',
                  member_id = member_id,
                  variable_id= var_id)
    print (cat.df)
    number_grid_label = np.unique (cat.df['grid_label']).shape[0]
    grid_label0 = np.unique(cat.df['grid_label'])[0]
    print ('Number of grid labels = %s' %number_grid_label)
    if number_grid_label > 1 :
        cat = col.search(source_id = CMIP6_model_id, 
                      experiment_id= 'historical', 
                      table_id='Omon',
                      variable_id= var_id, 
                      member_id = member_id,
                      grid_label = grid_label0)
        
    
    print ('Number of historical SST outputs = %s' %len(cat.df))
    
    ## Save nc files
    dset_dict = cat.to_dataset_dict(zarr_kwargs={'consolidated': True})
    list_keys = list(dset_dict.keys())
    
    for key in list_keys :
        ds = dset_dict [key].tos.squeeze()
        ds.to_netcdf ('Data/%s.%s.%s.nc' %(key, var_id, member_id))
   
        print ('Done')
        
    ## Create a repertory
    nc_files = glob.glob('Data/*MIP*%s*.nc' %CMIP6_model_id) 
    

    for filename in nc_files :
        print (filename)
        _, institute_id, source_id, experiment_id, table_id, grid_id, var_id, member_id, _ = filename.split(".")

        if not os.path.exists('Data/Data-%s/' %CMIP6_model_id):
            os.mkdir('Data/Data-%s/' %CMIP6_model_id)
        else:
            print("Directory already exists")
    

        savename = 'Data/Data-%s/processed_%s_2x2_%s_%s_%s_%s.nc' %(source_id, source_id, name_save, table_id, var_id, member_id)
        print (savename)
    
        if not os.path.exists(savename):
            # Create and execute the CDO commands
            commands = [
                f"cdo selyear,%s/%s %s output.nc" %(year1, year2, filename),
                f"cdo -L -remapbil,r180x90 output.nc output2.nc",
                f"cdo sellonlatbox,0,360,-60,60 -selname,%s output2.nc output3.nc" %var_id,
                f"cdo -L -ymonsub -selname,%s output3.nc -ymonmean -selname,%s output3.nc output4.nc" %(var_id, var_id),
                f"cdo detrend -selname,%s output4.nc %s" %(var_id, savename),
            ]
        
            for command in commands:
                execute_cdo_command(command)
        
            # Clean up the created files
            files_to_remove = ["output.nc", "output2.nc", "output3.nc", "output4.nc"]
            for file in files_to_remove:
                if os.path.isfile(file):
                    os.remove(file)
            
            # If you want to delete the original nc file
            os.remove(filename)
         
