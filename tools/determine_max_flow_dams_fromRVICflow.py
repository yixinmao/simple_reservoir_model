#!/usr/local/anaconda/bin/python

# This script determines maximum flow release at each dam, estimated from RVIC simulated flow

import csv
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime as dt
import sys
import my_functions

cfg = my_functions.read_config(sys.argv[1])

start_date = dt.datetime(cfg['OPTION']['start_date'][0], \
                         cfg['OPTION']['start_date'][1], \
                         cfg['OPTION']['start_date'][2], 12, 0)
end_date = dt.datetime(cfg['OPTION']['end_date'][0], \
                       cfg['OPTION']['end_date'][1], \
                       cfg['OPTION']['end_date'][2], 12, 0)

#================================================#
# Load dam info
#================================================#
df_dam_info = pd.read_csv(cfg['INPUT']['dam_to_model_info_path'])

#================================================#
# Load and process RVIC flow
#================================================#
# Load data
da_rvic = my_functions.read_RVIC_output(cfg['INPUT']['rvic_output_nc'], \
                                        output_format='grid', outlet_ind=-1)  # [cfs]
# Extract period of time to consider
da_rvic_to_consider = da_rvic.sel(time=slice(start_date, end_date))

#================================================#
# Process each dam
# 10% exceedence, daily data
#================================================#
#=== Loop over each dam ===#
for i in range(len(df_dam_info)):
    #=== Get dam info ===#
    lat = df_dam_info.ix[i, 'grid_lat']
    lon = df_dam_info.ix[i, 'grid_lon']
    dam_number = df_dam_info.ix[i, 'dam_number']
    dam_name = df_dam_info.ix[i, 'dam_name']
    print 'Processing dam {}...'.format(dam_number)

    #=== Extract RVIC flow for this dam location ===#
    s_rvic = da_rvic_to_consider.loc[:,lat,lon].to_series()
    
    #=== Determine max allowed flow ===#
    max_flow = my_functions.determine_max_flow_dam(s_rvic, cfg['OPTION']['exceedence'])

    #=== Save results in df ===#
    df_dam_info.ix[i,'max_flow_cfs'] = max_flow

#================================================#
# Save new dam info to file
#================================================#
df_dam_info['max_flow_cfs'] = df_dam_info['max_flow_cfs'].map(lambda x: '%.1f' % x)
df_dam_info.to_csv(cfg['OUTPUT']['output_dam_info_new_path'], index=False)




