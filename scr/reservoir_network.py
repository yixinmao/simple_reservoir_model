#!/usr/local/anaconda/bin/python

import sys
import numpy as np
import xray
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------------------------------------------------- #
def read_config(config_file, default_config=None):
    """
    Return a dictionary with subdictionaries of all configFile options/values
    """

    from netCDF4 import Dataset
    try:
        from cyordereddict import OrderedDict
    except:
        from collections import OrderedDict
    try:
        from configparser import SafeConfigParser
    except:
        from ConfigParser import SafeConfigParser
    import configobj

    config = SafeConfigParser()
    config.optionxform = str
    config.read(config_file)
    sections = config.sections()
    dict1 = OrderedDict()
    for section in sections:
        options = config.options(section)
        dict2 = OrderedDict()
        for option in options:
            dict2[option] = config_type(config.get(section, option))
        dict1[section] = dict2

    if default_config is not None:
        for name, section in dict1.items():
            if name in default_config.keys():
                for option, key in default_config[name].items():
                    if option not in section.keys():
                        dict1[name][option] = key

    return dict1
# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #
def config_type(value):
    """
    This function is originally from tonic (author: Joe Hamman); modified so that '\' is considered as an escapor. For example, '\,' can be used for strings with ','. e.g., Historical\, 1980s  will be recognized as one complete string
    Parse the type of the configuration file option.
    First see the value is a bool, then try float, finally return a string.
    """

    import cStringIO
    import csv

    val_list = [x.strip() for x in csv.reader(cStringIO.StringIO(value), delimiter=',', escapechar='\\').next()]
    if len(val_list) == 1:
        value = val_list[0]
        if value in ['true', 'True', 'TRUE', 'T']:
            return True
        elif value in ['false', 'False', 'FALSE', 'F']:
            return False
        elif value in ['none', 'None', 'NONE', '']:
            return None
        else:
            try:
                return int(value)
            except:
                pass
            try:
                return float(value)
            except:
                return value
    else:
        try:
            return list(map(int, val_list))
        except:
            pass
        try:
            return list(map(float, val_list))
        except:
            return val_list

# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #
def process_rule_curve(path, start_date, end_date):
    ''' This function reads in 365 days of rule curve storage for a reservoir, and prepare a rule curve time series for the whole time range desired

    Input:
        path: annual rule curve data path
              (365 days: <month> <day> <data [acre-feet]>, with a header line)
        start_date, end_date: start and end date for reservoir operation simulation [dt.datetime]

    Return:
        a pd.Series with dates as index and rule curve storage as data [acre-feet]

    '''

    import pandas as pd
    import numpy as np

    rule_curve_annual = np.loadtxt(path, skiprows=1)  # Load annual rule curve data
    s_rule_curve = pd.Series(index=pd.date_range(start_date,end_date))  # create empty Series

    # loop over each day of year
    for i in range(len(rule_curve_annual)):
        month = int(rule_curve_annual[i,0])
        day = int(rule_curve_annual[i,1])
        s_rule_curve.loc[(s_rule_curve.index.month==month) & (s_rule_curve.index.day==day)] = rule_curve_annual[i,2]
    s_rule_curve.loc[(s_rule_curve.index.month==2) & (s_rule_curve.index.day==29)] = rule_curve_annual[58,2] # set 2/29 equal to 2/28

    return s_rule_curve
# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #
def simulate_reservoir_operation(orig_flow, rule_curve, init_S, top_vol, bot_vol, max_flow, min_flow):
    ''' This function simulates reservoir operation and generates modified release flow;
        only simulates period of time specified by input rule curve time range

    Input:
        orig_flow: pd.Series of original flow, inflow to reservoir [cfs]
        rule_curve: pd.Series of rule curve storage [acre-feet]; can be shorter than orig_flow data; will be the time range that simulates reservoir operation
        init_S: initial storage in the reservoir [acre-feet]
        top_vol: top volumn of reservoir [acre-feet]
        bot_vol: bottom volumn of reservoir [acre-feet]
        max_flow: maximum allowed release [cfs]
        min_flow: minimum allowed release [cfs]

    Return:
        release, storage: pd.Series of flow release [cfs] and reservoir storage [acre-feet]
                          both only for the period defined by rule_curve

    Require:
        select_time_range(data, start_datetime, end_datetime)
    '''

    import pandas as pd

    #=== Convert units ===#
    top_vol = top_vol * 43560.0  # convert [acre-feet] to [ft3]
    bot_vol = bot_vol * 43560.0  # convert [acre-feet] to [ft3]
    init_S = init_S * 43560.0  # convert [acre-feet] to [ft3]
    rule_curve = rule_curve * 43560.0  # convert [acre-feet] to [ft3]

    #=== Specify date range to simulate and select out corresponding inflow data ===#
    start_date_to_run = rule_curve.index[0]
    end_date_to_run = rule_curve.index[-1]
    inflow = select_time_range(orig_flow, start_date_to_run, end_date_to_run)

    #=== Initialize ===#
    storage = pd.Series(index=pd.date_range(start_date_to_run,end_date_to_run))  # initialize storage ts [acre-feet]
    release = pd.Series(index=pd.date_range(start_date_to_run,end_date_to_run))  # initialize release ts [cfs]
    S = init_S

    #=== Loop over each day ===#
    for t in range(len(rule_curve)):
        # Maximum available water to release
        max_avail = S + inflow[t]*86400 - bot_vol  # [ft3/day]
        # Rease required to bring storage to rule curve
        rule_req = max(0, S + inflow[t]*86400 - rule_curve[t])  # [ft3/day]
        # Additional flood max capacity
        flood_cap = top_vol - rule_curve[t]  # [ft3]
        # Step 1 - preliminary release
        prelim_release = min(max_avail, max(rule_req, min_flow*86400))  # [ft3/day]
        # Step 2 - final release (check flood)
        reduced_release = max(max_flow*86400, prelim_release - flood_cap)  # [ft3/day]
        if prelim_release <= max_flow*86400:
            final_release = prelim_release
        else:
            final_release = reduced_release  # [ft3/day]
        release[t] = final_release / 86400.0  # convert to [cfs]
        # Update storage
        S = S + inflow[t]*86400 - final_release
        storage[t] = S / 43560.0  # convert [ft3] to [acre-feet]

    return release, storage
# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #
def find_downstream_grid(da_flowdir, lat, lon, dlatlon):
    ''' This function finds the immediate downstream grid cell based on 1-8 formatted flow direction
    Input:
        da_flowdir: xray.DataArray of flow direction file (1-8 format)
        lat, lon: lat and lon of the current grid cell
        dlatlon: delta lat and lon (e.g., 0.125)

    Return:
        lat, lon: lat lon of the immediate downstream grid cell
                  (if the current grid cell is an outlet, return -999, -999)

    '''

    import numpy as np

    #=== Extract flow direction of the current grid cell ===#
    flowdir = int(da_flowdir.loc[lat, lon].values)

    #=== Identify the next grid cell ===#
    lat_next, lon_next = lat, lon  # initialize
    if flowdir not in [1,2,3,4,5,6,7,8]:  # if don't have valid direction, then it's an outlet
        return -999, -999
    else:
        # Process lat
        if flowdir in [1,2,8]:
            lat_next = lat + dlatlon
        elif flowdir in [4,5,6]:
            lat_next = lat - dlatlon
        # Process lon
        if flowdir in [2,3,4]:
            lon_next = lon + dlatlon
        elif flowdir in [6,7,8]:
            lon_next = lon - dlatlon

    #=== Check if the next grid cell is still within basin ===#
    if da_flowdir.loc[lat_next,lon_next].values not in [1,2,3,4,5,6,7,8]:
    # if the next cell outside of basin
        return -999, -999
    else:  # if the next cell still within the basin
        return lat_next, lon_next
# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #
def modify_flow_all_downstream_cell(lat, lon, orig_flow, release, da_flow, dlatlon, da_flowdir, da_flowdis, velocity):
    ''' This function modifies flow at all downstream cells from an operated dam

    Input:
        lat, lon: lat and lon of the dam
        orig_flow: pd.Series of original inflow at the dam [cfs]
        release: pd.Series of simulated release at the dam [cfs]; only for period with operation
        da_flow: xray.DataArray of original streamflow field (time, lat, lon)
        dlatlon: delta lat and lon (e.g., 0.125)
        da_flowdir: xray.DataArray of flow direction file (1-8 format)
        da_flowdis: xray.DataArray of flow distance file [m]
        velocity: wave velocity [m/s]

    Return: xray.DataArray of modified streamflow field after this dam is operated
            (time, lat, lon)

    Require:
        find_downstream_grid(da_flowdir, lat, lon, dlatlon)

    '''

    import pandas as pd
    import datetime as dt

    #=== Find period with reservoir operation ===#
    start_date_to_run = release.index[0]
    end_date_to_run = release.index[-1]

    #=== Calculate delta flow ===#
    s_dflow = release - orig_flow[pd.date_range(start_date_to_run, end_date_to_run)]

    #=== Modify flow at the first grid cell (i.e., the dam grid cell) ===#
    da_flow.loc[pd.date_range(start_date_to_run, end_date_to_run),lat,lon] = release

    #=== Initialize flow distance from the dam to the current downstream grid cell [m] ===#
    fdis = 0

    #=== Move to the immediate downstream grid cell ===#
    lat_current, lon_current = find_downstream_grid(da_flowdir, lat, lon, dlatlon)
    if lat_current==-999 and lon_current==-999:
    # if the immediate downstream grid cell is outside of the basin,
    # done with all modification for this dam
        return da_flow

    #=== Loop over each downstream grid cell, and modify flow ===#
    while 1:
        #=== Identify time lag (to flow from dam to the current grid cell) ===#
        fdis = fdis + da_flowdis.loc[lat_current, lon_current]
        lag_days = int(round(fdis / velocity / 86400.0))  # [day], precise to integer days
        #=== Modify flow for the current grid cell ===#
        # Identify lagged time range
        start_date_lagged = start_date_to_run + dt.timedelta(days=lag_days)
        end_date_lagged = end_date_to_run + dt.timedelta(days=lag_days)
        # Modify flow for this grid cell
        if (end_date_lagged-orig_flow.index[-1]).days <= 0: # if within orig_flow time range
            da_flow.loc[pd.date_range(start_date_lagged, end_date_lagged),lat_current,lon_current] \
                    += s_dflow.values
        else:   # if later than orig_flow time range
            days_to_modify = (orig_flow.index[-1] - start_date_lagged).days + 1
            da_flow.loc[pd.date_range(start_date_lagged, orig_flow.index[-1]),\
                                      lat_current,lon_current] \
                    += s_dflow.values[0:days_to_modify]
        #=== Move to the next downstream grid cell ===#
        lat_current, lon_current = find_downstream_grid(da_flowdir, \
                                        lat_current, lon_current, dlatlon)
        if lat_current==-999 and lon_current==-999:
        # if the immediate downstream grid cell is outside of the basin,
        # done with all modification for this dam
            break
    return da_flow
# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #
def select_time_range(data, start_datetime, end_datetime):
    ''' This function selects out the part of data within a time range

    Input:
        data: [dataframe/Series] data with index of datetime
        start_datetime: [dt.datetime] start time
        end_datetime: [dt.datetime] end time

    Return:
        Selected data (same object type as input)
    '''

    import datetime as dt

    start = data.index.searchsorted(start_datetime)
    end = data.index.searchsorted(end_datetime)

    data_selected = data.ix[start:end+1]

    return data_selected
# -------------------------------------------------------------------- #



# Read in config file
cfg = read_config(sys.argv[1])

# Process dates
start_date_to_run = dt.datetime(cfg['PARAM']['start_date_to_run'][0], \
                                cfg['PARAM']['start_date_to_run'][1], \
                                cfg['PARAM']['start_date_to_run'][2], 12, 0)
end_date_to_run = dt.datetime(cfg['PARAM']['end_date_to_run'][0], \
                              cfg['PARAM']['end_date_to_run'][1], \
                              cfg['PARAM']['end_date_to_run'][2], 12, 0)

#====================================================================#
# Load dam information and network information
#====================================================================#
#=== Load dam info ===#
df_dam_info = pd.read_csv(cfg['DAM_INFO']['dam_info_csv'])
#=== Load network info ===@
ds_network = xray.open_dataset(cfg['NETWORK']['route_nc'])
da_flowdir = ds_network['Flow_Direction']
da_flowdis = ds_network['Flow_Distance']
velocity = cfg['NETWORK']['wave_velocity']  # wave velocity

#====================================================================#
# Simulate each dam and modify flow downstream
#====================================================================#
#=== Load original flow data (RVIC grid format) ===#
ds_rvic = xray.open_dataset(cfg['INPUT']['rvic_output_path'])
ds_rvic = ds_rvic.isel(time=slice(0,-1))   # delete last junk date
da_rvic_flow = ds_rvic['streamflow'] * pow(1000./25.4/12, 3)  # convert m3/s to cfs
da_flow = da_rvic_flow.copy()  # flow field, will be modified

#=== Loop over each dam ===#
# Dam list must be from upstream to downstream order!
for i in range(len(df_dam_info)):
    #=== Extract dam info ===#
    lat = df_dam_info.ix[i]['grid_lat']  # dam grid lat
    lon = df_dam_info.ix[i]['grid_lon']  # dam grid lon
    dam_number = df_dam_info.ix[i]['dam_number']  # dam number
    dam_name = df_dam_info.ix[i]['dam_name']  # dam name
    top_vol = df_dam_info.ix[i]['top_vol_acre_feet']  # reservoir top volumn [acre-feet]
    bot_vol = df_dam_info.ix[i]['bot_vol_acre_feet']  # reservoir bottom volumn [acre-feet]
    max_flow = df_dam_info.ix[i]['max_flow_cfs']  # max flow [cfs]
    min_flow = df_dam_info.ix[i]['min_flow_cfs']  # min flow [cfs]
    year_operated = df_dam_info.ix[i]['year_operated_start_of_Calendar_year']
                    # year operation started
    print 'Simulating dam {}...'.format(dam_number)
    #=== Load and process rule curve ===#
    rule_curve_filename = os.path.join(cfg['DAM_INFO']['rule_curve_dir'], \
                              'dam{}_{}.txt'.format(dam_number, dam_name.replace(' ', '_')))
    s_rule_curve = process_rule_curve(rule_curve_filename, \
                                    start_date_to_run, end_date_to_run) # [acre-feet]
    # If year start operation is after the period considered, truncate the time before operation
    s_rule_curve = s_rule_curve.truncate(before=dt.datetime(year_operated, 1, 1))
    if len(s_rule_curve)==0:  # if no period is operating, do not simulate
        continue
        
    #=== Extract original flow data from RVIC output ===#
    s_rvic_flow = da_flow.loc[:,lat,lon].to_series()
    #=== Simulate reservoir operation ===#
    init_S = s_rule_curve.ix[0]  # set initial storage to the rule curve value of the first day of simulation
    s_release, s_storage = simulate_reservoir_operation(s_rvic_flow, s_rule_curve, \
                                              init_S, top_vol, bot_vol, max_flow, min_flow)
    #=== Modify flow for all downstream grid cells ===#
    da_flow = modify_flow_all_downstream_cell(\
                        lat, lon, \
                        orig_flow=s_rvic_flow, \
                        release=s_release, da_flow=da_flow, dlatlon=cfg['NETWORK']['dlatlon'], \
                        da_flowdir=da_flowdir, da_flowdis=da_flowdis, velocity=velocity)    
    #=== Save storage and rule curve ===#
    df = pd.DataFrame()
    df['year'] = s_storage.index.year
    df['month'] = s_storage.index.month
    df['day'] = s_storage.index.day
    df['storage_acre_ft'] = s_storage.values
    df['rule_curve_acre_ft'] = s_rule_curve.values
    df[['year', 'month', 'day', 'storage_acre_ft', 'rule_curve_acre_ft']].\
            to_csv('{}.storage.dam{}.txt'.format(cfg['OUTPUT']['out_flow_basepath'], \
                                                 dam_number), \
            sep='\t', index=False)
    
#====================================================================#
# Save modified streamflow to netCDF file
#====================================================================#

#=== Save modified streamflow ===#
ds_flow_new = xray.Dataset({'streamflow': (['time', 'lat', 'lon'], da_flow.values)}, \
                           coords={'lat': (['lat'], ds_rvic['lat'].values), \
                                   'lon': (['lon'], ds_rvic['lon'].values), \
                                   'time': (['time'], da_flow['time'].values)})
ds_flow_new['streamflow'].attrs['units'] = 'cfs'
ds_flow_new['streamflow'].attrs['long_name'] = 'Simulated regulated streamflow'

ds_flow_new.to_netcdf('{}.modified_flow.nc'.format(cfg['OUTPUT']['out_flow_basepath']), \
                      format='NETCDF4_CLASSIC')
ds_flow_new.close()

#=== Save flow change before and after reservoir operation ===#
ds_flow_delta = xray.Dataset({'flow_delta': (['time', 'lat', 'lon'], \
                                             da_flow.values-da_rvic_flow.values)}, \
                             coords={'lat': (['lat'], ds_rvic['lat'].values), \
                                     'lon': (['lon'], ds_rvic['lon'].values), \
                                     'time': (['time'], da_flow['time'].values)})
ds_flow_delta['flow_delta'].attrs['units'] = 'cfs'
ds_flow_delta['flow_delta'].attrs['long_name'] = 'Simulated streamflow difference (regulated-unregulated'

ds_flow_delta.to_netcdf('{}.modified_delta_flow.nc'.format(cfg['OUTPUT']['out_flow_basepath']), \
                        format='NETCDF4_CLASSIC')
ds_flow_delta.close()


