# function categories:
# 1) pre-processing
# 2) quandl/quantcha

def tidyfy_surp_df(df):
    """Clean up Surp dataframes prior to join"""
    new_cols = ['ticker_symbol',
                 'co_name',
                 'unique_earnings_code',
                 'factset_sector_num',
                 'factset_ind_num',
                 'calendar_qtr',
                 'fiscal_qtr',
                 'adtv_prev_month',
                 'report_date',
                 'eps_est',
                 'eps_actual',
                 'surp_amt',
                 'rtn_t+3',
                 'mkt_t+3_rtn',
                 'rel_t+3_rtn',
                 'num_ests_qtr_end',
                 't-7_high_est',
                 't-7_low_est',
                 'est_spread',
                 'spread_adj_surp']
    
    tidy_df = df[new_cols]
    tidy_df.reset_index()
    return tidy_df

def create_feature_df_index(df):
    """Creates a new Index for extracting columns coded with a trailing 'F'from a dataframe"""
    
    old_columns = list(df.columns)
    new_columns = []
    
    for col in old_columns:
        if col[-1] == 'F':
            new_columns.append(col)
        
    new_columns.insert(0, 'unique_earnings_code')
    ind_obj = pd.Index(new_columns)
    
    return ind_obj


def clean_feature_bind(surp_df, feature_df, retained_columns):
    """Creates a tidied up df from a surp_df and feature_df, based on an Index object of cols to retain."""
    bound_df = pd.merge(surp_df, feature_df[retained_columns], on='unique_earnings_code')
    return bound_df

def write_merged_frames(surp_lst, features_lst):
    """Create combined dataframes from two lists of dataframe names: surp & features"""
    combined_df_lst = []
    for s_df, f_df in zip(surp_lst, features_lst):
        #create quarter tag
        tag = s_df[-8:]

        # read surp df
        surp_df = pd.read_csv('data/'+s_df)
        tidy_surp_df = tidyfy_surp_df(surp_df)
        # read feature df
        feature_df = pd.read_csv('data/'+f_df)

        # create list of columns to retain
        retained_cols = create_feature_df_index(feature_df)

        # create combined df
        combined_df = clean_feature_bind(tidy_surp_df, feature_df, retained_cols)

        # write combined_df to a csv file and store in data folder
        combined_df.to_csv('data/combined_'+tag)
        
        # record df written to a list
        combined_df_lst.append('combined_'+tag)
    
    return combined_df_lst



def stack_frames(sequence):
    sequence = sequence.copy()

    combined_full = pd.concat([pd.read_csv(f'data/{file}', low_memory=False) for file in sequence])
    combined_full = combined_full.drop(columns=['Unnamed: 0']).drop_duplicates()
    
    combined_full.to_csv('data/combined_full_set.csv')
















# create a helper function for getting a measure from the quantcha / quandl API

def get_vol_measure(ticker, date, measure='Hv90'):
    """Retrieves a value from the Quantcha Historical and Implied Volatility API for a given stock and date.
    
    INPUTS: 'Ticker', 'YYYY-MM-DD', and measure name.
    For reference, measure names are contained in the dictionary: vol_col_names
        
    OUPUT: A float
    """
    
    measure_value = (quandl.get('VOL/'+ticker,
                           start_date=date,
                           end_date=date, 
                           column_index=vol_col_names.get(measure)))
                
    return float(measure_value.values)


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
