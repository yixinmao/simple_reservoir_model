#!/usr/local/anaconda/bin/python

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

#==============================================================
#==============================================================

def read_USGS_data(file, columns, names):
	'''This function reads USGS streamflow from the directly downloaded format (date are in the 3rd columns)

	Input: 
		file: directly downloaded streamflow file path [str]
		columns: a list of data colomn numbers, starting from 1. E.g., if the USGS original data has three variables: max_flow, min_flow, mean_flow, and the desired variable is mean_flow, then columns = [3]
		names: a list of data column names. E.g., ['mean_flow']; must the same length as columns

	Return:
		a pd.DataFrame object with time as index and data columns (NaN for missing data points)

	Note: returned data and flow might not be continuous if there is missing data!!!

	'''

	import numpy as np
	import datetime as dt
	import pandas as pd

	ndata = len(columns)
	if ndata != len(names):  # check input validity
		print "Error: input arguments 'columns' and 'names' must have same length!"
		exit()

	f = open(file, 'r')
	date_array = []
	data = []
	for i in range(ndata):
		data.append([])
	while 1:
		line = f.readline().rstrip("\n")  # read in one line
		if line=="":
			break
		line_split = line.split('\t')
		if line_split[0]=='USGS':  # if data line
			date_string = line_split[2]  # read in date string
			date = dt.datetime.strptime(date_string, "%Y-%m-%d")  # convert date to dt object
			date_array.append(date)

			for i in range(ndata):  # for each desired data variable
				col = columns[i]
				if line_split[3+(col-1)*2] == '':  # if data is missing
					value = np.nan
				elif line_split[3+(col-1)*2] == 'Ice':  # if data is 'Ice'
					value = np.nan
				else:  # if data is not missing
					value = float(line_split[3+(col-1)*2])
				data[i].append(value)

	data = np.asarray(data).transpose()
	df = pd.DataFrame(data, index=date_array, columns=names)
	return df

#==============================================================
#==============================================================

def convert_YYYYMMDD_to_datetime(year, month, day):
	''' Convert arrays of year, month, day to datetime objects
	Input:
		year: an array of years
		month: an array of months
		day: an array of days
		(The three arrays must be the same length)
	Return:
		A list of datetime objects
'''

	import numpy as np
	import datetime as dt

	# Check if the input arrays are the same length
	if len(year)!=len(month) or len(year)!=len(day) or len(month)!=len(day):
		print "Error: the length of input date arrays not the same length!"
		exit()

	n = len(year)
	date = []
	for i in range(n):
		date.append(dt.datetime(year=np.int(np.round(year[i])), month=np.int(np.round(month[i])), day=np.int(np.round(day[i]))))

	return date

#==============================================================
#==============================================================

def convert_time_series_to_df(time, data, columns):
	'''This function converts datetime objects and data array to pandas dataframe object

	Input:
		time: a list of datetime objects, e.g. [dt.datetime(2011,1,1), dt.datetime(2011,1,3)]
		data: a 1-D or 2D array of corresponding data; if 2-D, should have the same number of rows as 'time' length
		columns: a list of column names, the same length as the number of columns of 'data', e.g. ['A', 'B', 'C']

	Return: a dataframe object
	'''

	import pandas as pd
	df = pd.DataFrame(data, index=time, columns=columns)
	return df

#==============================================================
#==============================================================

def convert_time_series_to_Series(time, data):
	'''This function converts datetime objects and data array to pandas Series object

	Input:
		time: a list of datetime objects, e.g. [dt.datetime(2011,1,1), dt.datetime(2011,1,3)]
		data: a 1-D array of corresponding data

	Return: a Series object
	'''

	import pandas as pd
	s = pd.Series(data, index=time)
	return s

#==============================================================
#==============================================================

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

#==============================================================
#==============================================================

def plot_date_format(ax, time_range=None, locator=None, time_format=None):
	''' This function formatting plots by plt.plot_date
	Input:
		ax: plotting axis
		time range: a tuple of two datetime objects indicating xlim. e.g., (dt.date(1991,1,1), dt.date(1992,12,31))

	'''

	import matplotlib.pyplot as plt
	import datetime as dt
	from matplotlib.dates import YearLocator, MonthLocator, DateFormatter

	# Plot time range
	if time_range!=None:
		plt.xlim(time_range[0], time_range[1])

	# Set time locator (interval)
	if locator!=None:
		if locator[0]=='year':
			ax.xaxis.set_major_locator(YearLocator(locator[1]))
		elif locator[0]=='month':
			ax.xaxis.set_major_locator(MonthLocator(interval=locator[1]))

	# Set time ticks format
	if time_format!=None:
		ax.xaxis.set_major_formatter(DateFormatter(time_format))

	return ax

#==============================================================
#==============================================================

def find_first_and_last_nonNaN_index(s):
	''' This function finds the first and last index with valid column values
	Input:
		df: Series object
		column: data column name to be investigated [str]
	'''

	import pandas as pd

	first_index = s[s.notnull()].index[0]
	last_index = s[s.notnull()].index[-1]
	return first_index, last_index

#==============================================================
#==============================================================

def find_data_common_range(list_s):
    ''' This function determines the common range of valid data of multiple datasets (missing data in the middle also count as 'valid')

    Input: a list of pd.Series objects

    Return: first and last datetimes of common range (Note: if one or all data have missing data at the beginning or end, those missing periods will be omitted; however, if there is missing data in the middle, those missing points will still be considered as valid)

    Requred:
        find_first_and_last_nonNaN_index
    '''

    list_first_date = []
    list_last_date = []
    for i in range(len(list_s)):
        s = list_s[i]
        s_first_date, s_last_date =  find_first_and_last_nonNaN_index(s)
        list_first_date.append(s_first_date)
        list_last_date.append(s_last_date)

    data_avai_start_date = sorted(list_first_date)[-1]
    data_avai_end_date = sorted(list_last_date)[0]

    return data_avai_start_date, data_avai_end_date

#==============================================================
#==============================================================

def find_full_water_years_within_a_range(dt1, dt2):
	''' This function determines the start and end date of full water years within a time range

	Input:
		dt1: time range starting time [dt.datetime]
		dt2: time range ending time [dt.datetime]

	Return:
		start and end date of full water years
	'''

	import datetime as dt

	if dt1.month <= 9:  # if dt1 is before Oct, start from Oct 1 this year
		start_date_WY = dt.datetime(dt1.year, 10, 1)
	elif dt1.month==10 and dt1.day==1:  # if dt1 is on Oct 1, start from this date
		start_date_WY = dt.datetime(dt1.year, 10, 1)
	else:  # if dt1 is after Oct 1, start from Oct 1 next year
		start_date_WY = dt.datetime(dt1.year+1, 10, 1)

	if dt2.month >=10:  # if dt2 is Oct or after, end at Sep 30 this year
		end_date_WY = dt.datetime(dt2.year, 9, 30)
	elif dt2.month==9 and dt2.day==30:  # if dt2 is on Sep 30, end at this date
		end_date_WY = dt.datetime(dt2.year, 9, 30)
	else:  # if dt2 is before Sep 30, end at Sep 30 last year
		end_date_WY = dt.datetime(dt2.year-1, 9, 30)

	if (end_date_WY-start_date_WY).days > 0:  # if at least one full water year
		return start_date_WY, end_date_WY
	else: # else, return -1 
		return -1

#==============================================================
#==============================================================

def calc_monthly_data(data):
	'''This function calculates monthly mean values

	Input: [DataFrame/Series] with index of time
	Return: a [DataFrame/Series] object, with monthly mean values (the same units as input data)
	'''

	import pandas as pd
	data_mon = data.resample("M", how='mean')
	return data_mon

#==============================================================
#==============================================================

def wateryear(calendar_date):
    if calendar_date.month >= 10:
        return calendar_date.year+1
    return calendar_date.year

def TVA_week(calendar_date):
    ''' Return TVA weekly data formatted week number (1-52); 
        Week starts from Jan 1; the 52th week has 8 or 9 days'''
    import datetime as dt
    day = (calendar_date - dt.datetime(calendar_date.year, 1, 1)).days + 1
    if day <= 364:
        week = (day-1) / 7 + 1
    else:
        week = 52
    return calendar_date.year, week

def calc_ts_stats_by_group(data, by, stat):
    '''This function calculates statistics of time series data grouped by year, month, etc

    Input:
        df: a [pd.DataFrame/Series] object, with index of time
        by: string of group by, (select from 'year' or 'month' or 'WY')
        stat: statistics to be calculated, (select from 'mean')
        (e.g., if want to calculate monthly mean seasonality (12 values), by='month' and stat='mean')

    Return:
        A [dateframe/Series] object, with group as index (e.g. 1-12 for 'month')

    Require:
        wateryear
        find_full_water_years_within_a_range(dt1, dt2)
        select_time_range(data, start_datetime, end_datetime)
    '''

    import pandas as pd
    import calendar
    import datetime as dt

    if by=='year':
        if stat=='mean':
            data_result = data.groupby(lambda x:x.year).mean()
    elif by=='month':
        if stat=='mean':
            data_result = data.groupby(lambda x:x.month).mean()
    elif by=='WY':  # group by water year
        # first, secelect out full water years
        start_date, end_date = find_full_water_years_within_a_range(data.index[0], data.index[-1])
        data_WY = select_time_range(data, start_date, end_date)
        # then, group by water year
        if stat=='mean':
            data_result = data_WY.groupby(lambda x:wateryear(x)).mean()
    elif by=='TVA_week':  # group by week in calendar year, TVA weekly data format
        if stat=='mean':
            data_result = data.groupby(lambda x:TVA_week(x)).mean()
        # Put weekly data on the middle day of week
        index_new = []
        for i in range(len(data_result)):  # Loop over every week
            year = data_result.index[i][0]  # extract year
            week = data_result.index[i][1]  # extract week number
            day = 4 + (week-1)*7  # middle day (day of year)
            if calendar.isleap(year) and week==52:
                day = day + 1
            date = dt.datetime(year,1,1) + dt.timedelta(days=day-1)
            index_new.append(date)
        data_result.index = index_new

    return data_result

#==============================================================
#==============================================================

def plot_format(ax, xtick_location=None, xtick_labels=None):
	'''This function formats plots by plt.plot

	Input:
		xtick_location: e.g. [1, 2, 3]
		xtick_labels: e.g. ['one', 'two', 'three']
	'''

	import matplotlib.pyplot as plt

	ax.set_xticks(xtick_location)
	ax.set_xticklabels(xtick_labels)

	return ax

#==============================================================
#==============================================================

def calc_annual_cumsum_water_year(time, data):
	'''This function calculates cumulative sum of data in each water year

	Input:
		time: corresponding datetime objects
		data: an array of data

	Return:
		time: the same as input
		data_cumsum: an array of annual (water year based) cumsum data; if there is missing data at a point, data_cumsum is np.nan after this point in this water year
	'''

	import numpy as np

	# Check if data and time is of the same length
	if len(data)!=len(time):
		print 'Error: data and time are not of the same length!'
		exit()

	data_cumsum = np.empty(len(data))
	for i in range(len(data)):
		if i==0:  # if the first day of record
			data_cumsum[0] = data[0]
		elif time[i].month!=10 or time[i].day!=1:  # if not Oct 1st, the same water year
			if (time[i]-time[i-1]).days>1:  # if the current day is not the next day of the previous day, i.e., if there is missing data here
				print 'Warning: missing data exists!'
				data_cumsum[i] = np.nan
			else:  # if no missing data at this time step, calculate cumsum
					data_cumsum[i] = data_cumsum[i-1] + data[i]
		else:  # if the current day is Oct 1st, and i!=0, the next water year
			data_cumsum[i] = data[i]

	return time, data_cumsum
 
#==============================================================
#==============================================================

def add_info_text_to_plot(fig, ax, model_info, stats, fontsize=14, bottom=0.3, text_location=-0.1):
	''' This function adds info text to the bottom of a plot 
		The text will include:
			Author; 
			Plotting date; 
			Model info (taking from input [str]);
			Stats (taking from input [str])
			bottom: the extent of adjusting the original plot; the higher the 'bottom', the more space it would be left for the texts
			text_location: the location of the text; the more negative, the lower the textG
'''

	import matplotlib.pyplot as plt
	import datetime as dt

	# adjust figure to leave some space at the bottom
	fig.subplots_adjust(bottom=bottom)

	# determine text content
	author = 'Yixin'
	today = dt.date.today()
	plot_date = today.strftime("%Y-%m-%d")
	text_to_add = 'Author: %s\nDate plotted: %s\nModel info: %s\nStats: %s\n' %(author, plot_date, model_info, stats)

	# add text
	plt.text(0, text_location, text_to_add, horizontalalignment='left',\
			verticalalignment='top', transform=ax.transAxes, fontsize=fontsize)

	return fig, ax

#==============================================================
#==============================================================

def read_Lohmann_route_daily_output(path):
    ''' This function reads Lohmann routing model daily output

    Input: daily output file path
    Return: a pd.Series object with datetime as index and flow[cfs] as data

    '''

    import pandas as pd
    import datetime as dt

    parse = lambda x: dt.datetime.strptime(x, '%Y %m %d')

    # load data
    df = pd.read_csv(path, delim_whitespace=True, parse_dates=[[0,1,2]], index_col=0, date_parser=parse, header=None)
    df = df.rename(columns={3:'flow'})
    # convert data to pd.Series
    s = df.ix[:,0]

    return s

#==============================================================
#==============================================================

def plot_time_series(plot_date, list_s_data, list_style, list_label, plot_start, plot_end, xlabel=None, ylabel=None, title=None, fontsize=16, legend_loc='lower right', time_locator=None, time_format='%Y/%m', xtick_location=None, xtick_labels=None, add_info_text=False, model_info=None, stats=None, show=False):
    ''' This function plots daily data time series

    Input:
        plot_date: True for plot_date, False for plot regular time series
        list_s_data: a list of pd.Series objects to be plotted
        list_style: a list of plotting style (e.g., ['b-', 'r--']); must be the same size as 'list_s_data'
        list_label: a list of plotting label (e.g., ['Scenario1', 'Scenario2']); must be the same size as 'list_s_data'
        xlabel: [str]
        ylabel: [str]
        title: [str]
        fontsize: for xlabe, ylabel and title [int]
        legend_loc: [str]
        plot_start, plot_end: if plot_date=True, [dt.datetime]; if plot_date=False, [float/int]
        time_locator: time locator on the plot; 'year' for year; 'month' for month. e.g., ('month', 3) for plot one tick every 3 months [tuple]
        time_format: [str]
        xtick_location: a list of xtick locations [list of float/int]
        xtick_labels: a list of xtick labels [list of str]; must be the same length as 'xtick_locations'
        add_info_text: True for adding info text at the bottom of the plot
        model_info, stats: descriptions added in the info text [str]
        show: True for showing the plot

    Require:
        plot_date_format
        add_info_text_to_plot(fig, ax, model_info, stats)
        plot_format
    '''

    import matplotlib.pyplot as plt

    # Check if list_s_data, list_style and list_label have the same length
    if len(list_s_data) !=len(list_style) or len(list_s_data)!=len(list_label):
        print 'Input list lengths are not the same!'
        exit()

    fig = plt.figure(figsize=(12,8))
    ax = plt.axes()
    for i in range(len(list_s_data)):
        if plot_date==True:  # if plot date
            plt.plot_date(list_s_data[i].index, list_s_data[i], list_style[i], label=list_label[i])
#            list_s_data[i].plot(style=list_style[i], label=list_label[i])
        else:  # if plot regular time series
            plt.plot(list_s_data[i].index, list_s_data[i], list_style[i], label=list_label[i])
    if xlabel:
        plt.xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        plt.ylabel(ylabel, fontsize=fontsize)
    if title:
        plt.title(title, fontsize=fontsize)
    # format plot
    leg = plt.legend(loc=legend_loc, frameon=True)
    leg.get_frame().set_alpha(0)
    if plot_date==True:  # if plot date
        plot_date_format(ax, time_range=(plot_start, plot_end), locator=time_locator, time_format='%Y/%m')
    else:  # if plot regular time series
        plt.xlim([plot_start, plot_end])
        if xtick_location:
            plot_format(ax, xtick_location=xtick_location, xtick_labels=xtick_labels)
    # add info text
    if add_info_text==True:
        add_info_text_to_plot(fig, ax, model_info, stats)

    if show==True:
        plt.show()

    return fig

#==============================================================
#==============================================================

def plot_xy(list_x, list_y, list_style, list_label, figsize=(12,8), xlog=False, ylog=False, xlim=None, ylim=None, xlabel=None, ylabel=None, title=None, fontsize=16, legend_loc='lower right', add_info_text=False, model_info=None, stats=None, show=False):
	''' This function plots xy data

	Input:
		list_x, list_y: a list of x and y data; [np.array or list]
		list_style: a list of plotting style (e.g., ['b-', 'r--']); must be the same size as 'list_s_data'
		list_label: a list of plotting label (e.g., ['Scenario1', 'Scenario2']); must be the same size as 'list_s_data'
		xlog, ylog: True for plotting log scale for the axis; False for plotting regular scale
		xlabel: [str]
		ylabel: [str]
		title: [str]
		fontsize: for xlabe, ylabel and title [int]
		legend_loc: [str]
		xlim, ylim
		add_info_text: True for adding info text at the bottom of the plot
		model_info, stats: descriptions added in the info text [str]
		show: True for showing the plot

	Require:
		add_info_text_to_plot(fig, ax, model_info, stats)
		plot_format
	'''

	import matplotlib.pyplot as plt

	# Check if list_s_data, list_style and list_label have the same length
	if len(list_x) !=len(list_y) or len(list_x)!=len(list_style) or len(list_x)!=len(list_label):
		print 'Input list lengths are not the same!'
		exit()

	fig = plt.figure(figsize=figsize)
	ax = plt.axes()
	for i in range(len(list_x)):
		plt.plot(list_x[i], list_y[i], list_style[i], label=list_label[i])
	if xlabel:
		plt.xlabel(xlabel, fontsize=fontsize)
	if ylabel:
		plt.ylabel(ylabel, fontsize=fontsize)
	if title:
		plt.title(title, fontsize=fontsize)
	if xlog:
		ax.set_xscale('log')
	if ylog:
		ax.set_yscale('log')
	# format plot
	leg = plt.legend(loc=legend_loc, frameon=True)
	leg.get_frame().set_alpha(0)
	if xlim:
		plt.xlim(xlim)
	if ylim:
		plt.ylim(ylim)
	# add info text
	if add_info_text==True:
		add_info_text_to_plot(fig, ax, model_info, stats)

	if show==True:
		plt.show()

	return fig

#==============================================================
#==============================================================

def plot_monthly_data(list_s_data, list_style, list_label, plot_start, plot_end, xlabel=None, ylabel=None, title=None, fontsize=16, legend_loc='lower right', time_locator=None, time_format='%Y/%m', add_info_text=False, model_info=None, stats=None, show=False):
	''' This function plots monthly mean data time series

	Require:
		plot_date_format
		add_info_text_to_plot(fig, ax, model_info, stats)
		plot_time_series
		calc_monthly_data
	'''

	# Check if list_s_data, list_style and list_label have the same length
	if len(list_s_data) !=len(list_style) or len(list_s_data)!=len(list_label):
		print 'Input list lengths are not the same!'
		exit()

	# Calculate monthly mean data
	list_s_month = []   # list of new monthly mean data in pd.Series type 
	for i in range(len(list_s_data)):
		s_month = calc_monthly_data(list_s_data[i])
		list_s_month.append(s_month)

	# plot
	fig = plot_time_series(True, list_s_month, list_style, list_label, plot_start, plot_end, xlabel, ylabel, title, fontsize, legend_loc, time_locator, time_format, add_info_text=add_info_text, model_info=model_info, stats=stats, show=show)

	return fig

#==============================================================
#==============================================================

def plot_seasonality_data(list_s_data, list_style, list_label, plot_start, plot_end, xlabel=None, ylabel=None, title=None, fontsize=16, legend_loc='lower right', xtick_location=None, xtick_labels=None, add_info_text=False, model_info=None, stats=None, show=False):
	''' This function plots seasonality data time series (12 month's mean)

	Require:
		plot_date_format
		add_info_text_to_plot(fig, ax, model_info, stats)
		plot_time_series
		calc_ts_stats_by_group
		plot_format
	'''

	# Check if list_s_data, list_style and list_label have the same length
	if len(list_s_data) !=len(list_style) or len(list_s_data)!=len(list_label):
		print 'Input list lengths are not the same!'
		exit()

	# Calculate monthly mean data
	list_s_seas = []   # list of new monthly mean data in pd.Series type 
	for i in range(len(list_s_data)):
		s_seas = calc_ts_stats_by_group(list_s_data[i], 'month', 'mean') # index is 1-12 (month)
		list_s_seas.append(s_seas)

	# plot
	fig = plot_time_series(False, list_s_seas, list_style, list_label, plot_start, plot_end, xlabel, ylabel, title, fontsize, legend_loc, xtick_location=xtick_location, xtick_labels=xtick_labels, add_info_text=add_info_text, model_info=model_info, stats=stats, show=show)

	return fig

#==============================================================
#==============================================================

def plot_WY_mean_data(list_s_data, list_style, list_label, plot_start, plot_end, xlabel=None, ylabel=None, title=None, fontsize=16, legend_loc='lower right', time_locator=None, time_format='%Y/%m', add_info_text=False, model_info=None, stats=None, show=False):
	''' This function plots water year annual mean data time series

	Require:
		plot_date_format
		add_info_text_to_plot(fig, ax, model_info, stats)
		plot_time_series
		calc_ts_stats_by_group
	'''

	# Check if list_s_data, list_style and list_label have the same length
	if len(list_s_data) !=len(list_style) or len(list_s_data)!=len(list_label):
		print 'Input list lengths are not the same!'
		exit()

	# Calculate monthly mean data
	list_s_WY = []   # list of new monthly mean data in pd.Series type 
	for i in range(len(list_s_data)):
		s_WY = calc_ts_stats_by_group(list_s_data[i], 'WY', 'mean')
		list_s_WY.append(s_WY)

	# plot
	fig = plot_time_series(False, list_s_WY, list_style, list_label, plot_start, plot_end, xlabel, ylabel, title, fontsize, legend_loc, add_info_text=add_info_text, model_info=model_info, stats=stats, show=show)

	return fig

#==============================================================
#==============================================================

def calc_WY_mean(s_data):
	''' This function calculates mean annual data (water year) 
	
	Require:
		calc_ts_stats_by_group
	'''

	s_WY = calc_ts_stats_by_group(s_data, 'WY', 'mean')

	return s_WY

#==============================================================
#==============================================================

def plot_boxplot(list_data, list_xlabels, color_list, xlabel=None, rotation=0, ylabel=None, title=None, fontsize=16, legend_text_list=None, legend_color_list=None, legend_loc=None, add_info_text=False, model_info=None, stats=None, bottom=0.3, text_location=-0.1, show=False):
	''' This function plots a vertical boxplot

	Input:
		list_data: a list of boxplot groups to be plotted; each group is a list of data arrays
		list_xlabels: a list of xaxis labels correspondong to each group in list_data; must be the same length with list_data
		color_list: a list of colors for each box plot group; each element of this list corresponds to one box group, which is a list of colors of each box in this group; color_list must be the same length with list_data
		legend_text_list: a list of legend text to be added
		legend_color_list: a list of legend color to be added (must be the same length as legend_text_list)
		rotation: rotation angle (deg) of x labels

	Require:
		setBoxColors(bp, color_list)
	'''

	import matplotlib.pyplot as plt

	# Check if list_data and list_xaxis have the same length
	if len(list_data) !=len(list_xlabels):
		print 'Input list lengths are not the same!'
		exit()

	fig = plt.figure(figsize=(12,8))
	ax = plt.axes()
	position_count = 0
	position_label = []  # this records where xticks should be located (the first location of each group)
	for i in range(len(list_data)):  # plot each box group
		# Determine plotting box position
		position_label.append(position_count)
		position = []
		for j in range(len(list_data[i])):
			position.append(position_count)
			position_count = position_count + 1
		position_count = position_count + 1  # this is for some space between groups
		# Plot this box group
		bp = plt.boxplot(list_data[i], positions=position, widths=0.6)
		setBoxColors(bp, color_list[i])  # set color
		# Plot mean value as a dot
		for j in range(len(list_data[i])):
			plt.plot(position[j], list_data[i][j].mean(), marker='o', markerfacecolor=color_list[i][j], markeredgecolor=color_list[i][j])
	plt.xticks(position_label, list_xlabels, rotation=rotation)
	plt.xlim([position_label[0]-1, position_label[-1]+2])
	if xlabel:
		plt.xlabel(xlabel, fontsize=fontsize)
	if ylabel:
		plt.ylabel(ylabel, fontsize=fontsize)
	if title:
		plt.title(title, fontsize=fontsize)
	# add legend (draw temporary color lines and use them to create a legend)
	if legend_text_list!=None:
		plt_handles = []
		for i in range(len(legend_text_list)):
			h, = plt.plot(list_data[0][0].mean(), legend_color_list[i])
			plt_handles.append(h)
		plt.legend(plt_handles, legend_text_list, loc=legend_loc)
		for h in plt_handles:
			h.set_visible(False)
	# add info text
	if add_info_text==True:
		add_info_text_to_plot(fig, ax, model_info, stats, bottom=bottom, text_location=text_location)

	if show==True:
		plt.show()

	return fig

#==============================================================
#==============================================================

def setBoxColors(bp, color_list):
    ''' This function set color for each box on a boxplot

    Input:
        bp: a boxplot object (bp = plt.boxplot(...))
        color_list: a list of color; same length as number of boxes (e.g., ['b', 'r'])
    '''

    import matplotlib.pyplot as plt
    nbox = len(bp['boxes'])
    for i in range(nbox):
        plt.setp(bp['boxes'][i], color=color_list[i])
        plt.setp(bp['caps'][i*2], color=color_list[i])
        plt.setp(bp['caps'][i*2+1], color=color_list[i])
        plt.setp(bp['whiskers'][i*2], color=color_list[i])
        plt.setp(bp['whiskers'][i*2+1], color=color_list[i])
        plt.setp(bp['fliers'][i], color=color_list[i])
        plt.setp(bp['medians'][i], color=color_list[i])

#========================================================================
#========================================================================

def read_nc(infile, varname, dimension=-1, is_time=0):
	'''Read a variable from a netCDF file

	Input:
		input file path
		variable name
		dimension: if < 0, read in all dimensions of the variable; if >= 0, only read in the [dimension]th of the variable (index starts from 0). For example, if the first dimension of the variable is time, and if dimension=2, then only reads in the 3rd time step.
		is_time: if the desired variable is time (1 for time; 0 for not time). If it is time, return an array of datetime object

	Return:
		var: a numpy array of
	'''

	from netCDF4 import Dataset
	from netCDF4 import num2date

	nc = Dataset(infile, 'r')
	if is_time==0:  # if not time variable
		if dimension<0:
			var = nc.variables[varname][:]
		else:
			var = nc.variables[varname][dimension]
	if is_time==1:  # if time variable
		time = nc.variables[varname]
		if hasattr(time, 'calendar'):  # if time variable has 'calendar' attribute
			if dimension<0:
				var = num2date(time[:], time.units, time.calendar)
			else:
				var = num2date(time[dimension], time.units, time.calendar)
		else:  # if time variable does not have 'calendar' attribute
			if dimension<0:
				var = num2date(time[:], time.units)
			else:
				var = num2date(time[dimension], time.units)
	nc.close()
	return var

#========================================================================
#========================================================================

def get_nc_ts(infile, varname, timename):
	''' This function reads in a time series form a netCDF file
		(only suitable if the variable only has time dimension, or its other dimensions (e.g. lat and lon) are 1)

	Require:
		read_nc

	Input:
		infile: nc file path [string]
		varname: data variable name in the nc file [string]
		timename: time variable name in the nc file [string]

	Return:
		s: [pd.Series] object with index of time
	'''

	import pandas as pd

	data = read_nc(infile, varname, dimension=-1, is_time=0)
	time = read_nc(infile, timename, dimension=-1, is_time=1)
	data = data.squeeze()  # delete single-dimensions (e.g. lat=1, lon=1)

	s = pd.Series(data, index=time)
	return s

#==============================================================
#==============================================================

def plot_duration_curve(list_s_data, list_style, list_label, figsize=(10,10), xlog=False, ylog=False, xlim=None, ylim=None, xlabel=None, ylabel=None, title=None, fontsize=16, legend_loc='lower right', add_info_text=False, model_info=None, stats=None, show=False):
	''' This function plots duration curve for time series (exceedence is calculated by Weibull plotting position method)

	Require:
#		plot_date_format
#		add_info_text_to_plot(fig, ax, model_info, stats)
#		plot_time_series
#		plot_format
	'''

	import numpy as np

	# Check if list_s_data, list_style and list_label have the same length
	if len(list_s_data) !=len(list_style) or len(list_s_data)!=len(list_label):
		print 'Input list lengths are not the same!'
		exit()

	# Calculate
	list_data_sorted = []   # list of sorted data
	list_data_exceed = []   # list of exceedence for each data set
	for i in range(len(list_s_data)):
		data = list_s_data[i].values  # retrieve data as np array
		ndata = len(data)  # length of data
		data_exceed = np.empty(ndata)  # exceedence
		data_sorted = sorted(data, reverse=True, key=float)
		for j in range(ndata):
			data_exceed[j] = float((j+1)) / (ndata+1)
		list_data_sorted.append(data_sorted)
		list_data_exceed.append(data_exceed)

	# plot
	fig = plot_xy(list_data_exceed, list_data_sorted, list_style, list_label, \
				figsize=figsize, xlim=xlim, ylim=ylim, xlog=xlog, ylog=ylog, \
				xlabel=xlabel, ylabel=ylabel, title=title, fontsize=fontsize, \
				legend_loc=legend_loc, add_info_text=add_info_text, \
				model_info=model_info, stats=stats, show=show)

	return fig

#==============================================================
#==============================================================

def read_RMB_formatted_output(path, var='Tstream'):
    ''' This function reads formatted RBM output

    Input: 
        path - formatted RBM output path for a stream segment (columns: year; month; day; flow(cfs); T_stream(degC); Thead; Tair)
        var - variable needed; choose from: 'Tstream'; 'flow'; default: 'Tstream'
    Return: a pd.Series object with datetime as index, and T_stream as data

    '''

    import pandas as pd
    import datetime as dt

    parse = lambda x: dt.datetime.strptime(x, '%Y %m %d')

    # load data
    df = pd.read_csv(path, delim_whitespace=True, parse_dates=[[0,1,2]], index_col=0, date_parser=parse, header=None)
    df = df.rename(columns={3:'flow', 4:'Tstream', 5:'Thead', 6:'Tair'})
    # convert data to pd.Series
    if var=='Tstream': 
        s = df.ix[:,1]
    elif var=='flow':
        s = df.ix[:,0]
    else:
        print 'Error: unsupported RBM output variable!'
        exit()

    return s

#========================================================================
#========================================================================

def read_RVIC_output(filepath, output_format='array', outlet_ind=-1):
    ''' This function reads RVIC output netCDF file

    Input:
        filepath: path of the output netCDF file
        output_format: 'array' or 'grid' (currently only support 'array')
        outlet_ind: for 'array' format: index of the outlet to be read (index starts from 0); -1 for reading all outlets;
                    for 'grid' format: a tuple of grid cell lat and lon (e.g., (35.5, -120.5); -1 for reading all outlets)

    Return:
        If 'array' format:
            df - a DataFrame containing streamflow [unit: cfs]; column name(s): outlet name
            dict_outlet - a dictionary with outlet name as keys; [lat lon] as content
        If 'grid' format:
            a pd.Series of streamflow [unit: cfs] if only one grid cell;
            a xray.DataArray of streamflow [unit: cfs] if all grid cells

    '''

    import numpy as np
    import pandas as pd
    import xray

    ds = xray.open_dataset(filepath)

    if output_format=='array':
        #=== Read in outlet names ===#
        outlet_names = []
        for i, name in enumerate(ds['outlet_name']):
            outlet_names.append(str(name.values))

        #=== Read in outlet lat lon ===#
        dict_outlet = {}
        # If read all outlets
        if outlet_ind==-1:
            for i, name in enumerate(outlet_names):
                dict_outlet[name] = [ds['lat'].values[i], ds['lon'].values[i]]
        # If read one outlet
        else:
            dict_outlet[outlet_names[outlet_ind]] = \
                        [ds['lat'].values[outlet_ind], ds['lon'].values[outlet_ind]]

        #=== Read in streamflow variable ===#
        flow = ds['streamflow'].values
        flow = flow * np.power(1000/25.4/12, 3)  # convert m3/s to cfs
        # If read all outlets
        if outlet_ind==-1:
            df = pd.DataFrame(flow, index=ds.coords['time'].values, columns=outlet_names)
        # If read one outlet
        else:
            df = pd.DataFrame(flow[:,outlet_ind], index=ds.coords['time'].values, \
                              columns=[outlet_names[outlet_ind]])
    
        #=== Read in streamflow variable ===#
        df = df[:-1]  # Get rid of the last junk time step

        return df, dict_outlet

    elif output_format=='grid':
        ds = ds.isel(time=slice(0,-1))  # get rid of the last junk time step
        if outlet_ind==-1:
            return ds['streamflow'] * np.power(1000/25.4/12, 3)  # convert m3/s to cfs
        else:
            # Extract lat lon of grid cell desired
            lat = outlet_ind[0]
            lon = outlet_ind[1]
            return ds['streamflow'].loc[:,lat,lon].to_series() * np.power(1000/25.4/12, 3)  
                                                                    # convert m3/s to cfs

#========================================================================
#========================================================================

def kge(sim, obs):
    ''' Calculate Kling-Gupta Efficiency (function from Oriana) '''

    import numpy as np

#    obs_mask=obs/obs
#    sim *= obs_mask
#    obs *= obs_mask
    
    std_sim = np.std(sim)
    std_obs = np.std(obs)
    mean_sim = sim.mean(axis=0)
    mean_obs = obs.mean(axis=0)
    r_array = np.corrcoef(sim.values, obs.values)
    r = r_array[0,1]
    relvar = std_sim/std_obs
    bias = mean_sim/mean_obs
    kge = 1-np.sqrt(np.square(r-1) + np.square(relvar-1) + np.square(bias-1))
    return kge

#==============================================================
#==============================================================

def plot_scatter(list_x, list_y, list_s, list_c, list_marker, list_label, figsize=(12,8), linewidths=None, alpha=1, xlog=False, ylog=False, xlim=None, ylim=None, xlabel=None, ylabel=None, title=None, fontsize=16, legend_loc='lower right', add_info_text=False, model_info=None, stats=None, show=False):
    ''' This function plots xy data

    Input:
        list_x, list_y: a list of x and y data; [np.array or list]
        list_s, list_c, list_marker: a list of size, color and marker
        list_label: a list of plotting label (e.g., ['Scenario1', 'Scenario2']); must be the same size as 'list_s_data'
        xlog, ylog: True for plotting log scale for the axis; False for plotting regular scale
        xlabel: [str]
        ylabel: [str]
        title: [str]
        fontsize: for xlabe, ylabel and title [int]
        linewidths: width of marker line
        alpha: transparency; 0 for transparent; 1 for opaque
        legend_loc: [str]
        xlim, ylim
        add_info_text: True for adding info text at the bottom of the plot
        model_info, stats: descriptions added in the info text [str]
        show: True for showing the plot

    Require:
        add_info_text_to_plot(fig, ax, model_info, stats)
        plot_format
    '''

    import matplotlib.pyplot as plt

    # Check if list_s_data, list_style and list_label have the same length
    if len(list_x) !=len(list_y) or len(list_x)!=len(list_s) or len(list_x)!=len(list_c) or len(list_x)!=len(list_marker) or len(list_x)!=len(list_label):
        print 'Input list lengths are not the same!'
        exit()

    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    for i in range(len(list_x)):
        plt.scatter(list_x[i], list_y[i], s=list_s[i], c=list_c[i], \
                    marker=list_marker[i], linewidths=linewidths, \
                    alpha=alpha, label=list_label[i])
    if xlabel:
        plt.xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        plt.ylabel(ylabel, fontsize=fontsize)
    if title:
        plt.title(title, fontsize=fontsize)
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    # format plot
    leg = plt.legend(loc=legend_loc, frameon=True)
    leg.get_frame().set_alpha(0)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    # add info text
    if add_info_text==True:
        add_info_text_to_plot(fig, ax, model_info, stats)

    if show==True:
        plt.show()

    return fig

#========================================================================
#========================================================================

def define_map_projection(projection='gall', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180, resolution='i', land_color='grey', ocean_color='lightblue', lakes=True):
    '''Define projected map

    Return: the projection
    '''

    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt

    m = Basemap(projection=projection, llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, resolution=resolution)
    m.drawlsmask(land_color=land_color, ocean_color=ocean_color, lakes=lakes)
    m.drawcoastlines(linewidth=0.75)
    m.drawstates(linewidth=0.5)
    m.drawcountries(linewidth=0.5)

    return m

#========================================================================
#========================================================================

def mesh_xyz(lat, lon, data, dlat, dlon):
    ''' Convert xyz data into meshgrid format

    Input arguments: lat, lon, data (np array or list)
    Return: lat_mesh, lon_mesh, data_mesh_masked (no-data grids is masked)
    '''

    import numpy as np

    # Make meshed lon & lat
    min_lon = min(lon)  # min longitude
    max_lon = max(lon)  # max longitude
    nlon = int(round((max_lon-min_lon)/dlon + 1))  # number of longitude meshes
    min_lat = min(lat)  # min latitude
    max_lat = max(lat)  # max latitude
    nlat = int(round((max_lat-min_lat)/dlat + 1))  # number of latitude meshes

    lat_array = np.arange(max_lat, min_lat-dlat/2.0, -dlat)
    lon_array = np.arange(min_lon, max_lon+dlon/2.0, dlon)
    lon_mesh, lat_mesh = np.meshgrid(lon_array, lat_array)

    # Make meshed data
    data_mesh = np.empty((nlat, nlon))
    data_mesh[:] = np.nan
    for i in range(len(data)):
        lat_ind = int(round((max_lat-lat[i])/dlat))
        lon_ind = int(round((lon[i]-min_lon)/dlon))
        data_mesh[lat_ind][lon_ind] = data[i]
    data_mesh_masked = np.ma.masked_invalid(data_mesh)

    return lat_mesh, lon_mesh, data_mesh_masked

#========================================================================
#========================================================================

def kge_component(sim, obs):
    ''' Calculate Kling-Gupta Efficiency (function from Oriana) 
        This function returns each of the three components of KGE
        (bias, variance, covariance, all normalized to around zero)'''

    import numpy as np

    std_sim = np.std(sim)
    std_obs = np.std(obs)
    mean_sim = sim.mean(axis=0)
    mean_obs = obs.mean(axis=0)
    r_array = np.corrcoef(sim.values, obs.values)
    r = r_array[0,1]
    relvar = std_sim/std_obs
    bias = mean_sim/mean_obs
    kge = 1-np.sqrt(np.square(r-1) + np.square(relvar-1) + np.square(bias-1))
    return bias-1, relvar-1, r-1

#========================================================================
#========================================================================

def determine_max_flow_dam(s_flow, exceedence):
    ''' This function determines maximum allowed flow release at a dam, based exceedence on daily data

    Input:
        s_flow: pd.Series of daily flow data (during reservoir operation time period) [cfs]
        exceedence: exceedence (based on daily data) for determining max flow

    '''

    from scipy.interpolate import interp1d
    import numpy as np

    #=== Calculate exceedence for flow data ===#
    ndata = len(s_flow)  # length of data
    data_exceed = np.empty(ndata)  # exceedence
    data_sorted = sorted(s_flow.values, reverse=True, key=float)
    for j in range(ndata):
        data_exceed[j] = float((j+1)) / (ndata+1)

    #=== Interpolate to the exceedence value wanted ===#
    f = interp1d(data_exceed, data_sorted)
    return float(f(exceedence))
    




