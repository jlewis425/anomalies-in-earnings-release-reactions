# import required packages

import pandas as pd
import numpy as np
import quandl
from datetime import datetime, timedelta
import json




# quandl authentication
quandl.ApiConfig.api_key = "jUKZLy5xi7gGFf3sSF-r"

# load dictionary of column_name:column_index pairs for Quantcha Volatility data
with open('data/vol_col_names.json') as json_data:
    vol_col_names = json.load(json_data)

# assemble surprise metrics to create labels:

# read in initial file to start surp_data df
surp_data = pd.read_csv('data/surp_1q14.csv')

# create list of files to concatenate to base file (surp_1q14.csv)
additional_files = (['surp_2q14.csv',
                     'surp_3q14.csv',
                     'surp_4q14.csv',
                     'surp_1q15.csv',
                     'surp_2q15.csv',
                     'surp_3q15.csv',
                     'surp_4q15.csv',
                     'surp_1q16.csv',
                     'surp_2q16.csv',
                     'surp_3q16.csv',
                     'surp_4q16.csv',
                     'surp_1q17.csv',
                     'surp_2q17.csv',
                     'surp_3q17.csv',
                     'surp_4q17.csv',
                     'surp_1q18.csv',
                     'surp_2q18.csv'])


# loop through list and concatenate additional files to surp_data
for file in additional_files:
    temp = pd.read_csv('data/'+str(file))
    surp_data = pd.concat([surp_data, temp], sort=False)


# test whether observations have more than 2 analyst estimates
more_than_two_ests = surp_data['num_ests_qtr_end'] > 2

# eliminate observations with 2 or fewer estimates
surp_data = surp_data[more_than_two_ests]

# test whether rtn_t+3 is avaialble --> not NaN
rtn_avail = surp_data["rtn_t+3"].notna()

# eliminate observations where rtn_t+3 is not available
surp_data = surp_data[rtn_avail]
