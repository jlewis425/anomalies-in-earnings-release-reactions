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

# create a generalized measure delta function

def measure_delta(ticker, date, look_back_days, measure='Hv90'):
    """Provides a relative comparison for a given stock ticker and date, based on an arbitrary look_back_window,
    for any of 64 data itmes from the Quantcha Historical and Implied Volatility API.
    
    INPUTS: 'Ticker', 'YYYY-MM-DD', lookback days (as integer), and measure name.
    For reference, measure names are contained in the dictionary: vol_col_names
    ***NOTE: For best results, utilize a lookback window that is a multiple of 7 days from the given date.
    
    OUPUT: A float, based on the forumla: <measure at date> / <measure at date - lookback days>
    """
    
    date = datetime.strptime(date, '%Y-%m-%d')
    
    lookback_date = date - timedelta(days=int(look_back_days))
        
    date = str(date)
    date = date[0:10]
    
    lookback_date = str(lookback_date)
    lookback_date = lookback_date[0:10]
    
      
    curr_vol = (quandl.get('VOL/'+ticker,
                           start_date=date,
                           end_date=date, 
                           column_index=vol_col_names.get(measure)))
                
    
    prior_vol = (quandl.get('VOL/'+ticker,
                            start_date=lookback_date,
                            end_date=lookback_date,
                            column_index=vol_col_names.get(measure)))
    
    
    
    output = curr_vol.values / prior_vol.values
    return float(output)


