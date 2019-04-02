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


