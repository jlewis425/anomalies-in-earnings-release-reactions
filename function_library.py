# required packages
import pandas as pd
import numpy as np
import quandl
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import seaborn as sns


from dateutil.parser import parse
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss, precision_score, confusion_matrix, precision_recall_curve
from sklearn.metrics import roc_auc_score, f1_score, make_scorer, recall_score, average_precision_score

from sklearn.utils.multiclass import unique_labels
from sklearn.utils.fixes import signature
from sklearn.preprocessing import OneHotEncoder, LabelEncoder




# function categories:
# 1) pre-processing & data partitioning
# 2) model fitting & prediction
# 3) plotting
# 4) sequential simulation

# pre-processing & data partitioning functions
# helper functions

def _create_feature_df_index(df):
    """HELPER to create a new Index for extracting columns coded with a trailing 'F' from a dataframe"""
    
    old_columns = list(df.columns)
    new_columns = []
    
    for col in old_columns:
        if col[-1] == 'F':
            new_columns.append(col)
        
    new_columns.insert(0, 'unique_earnings_code')
    #new_columns.insert(3, 't-7')
    ind_obj = pd.Index(new_columns)
    
    return ind_obj

def _sector_rel_earn_yld(df):
    """HELPER to create sector_rel_earn_yield Feature and add it to the surprise df."""
    df = df.dropna(subset=['factset_sector_num'])
    sectors = df['factset_sector_num']
    ylds = df['qtr_end_eps_yld']
    temp = pd.concat([sectors, ylds], axis=1)
    new_col = temp.groupby('factset_sector_num').transform(lambda x: x - x.mean())
    new_col.rename(columns={'qtr_end_eps_yld':'sec_rel_eps_yld_F'}, inplace=True)
    df = pd.concat([df, new_col], axis = 1)
    
    return df

def _rel_earn_yld(df):
    """HELPER function to generate new rel_earn_yld Feature col and add it to the surprise df."""
    ylds = df['qtr_end_eps_yld']
    avg_yld = df['qtr_end_eps_yld'].mean()
    new_ylds = ylds - avg_yld
    new_col = pd.DataFrame(new_ylds)
    new_col.rename(columns={'qtr_end_eps_yld':'rel_eps_yld_F'}, inplace=True)
    df = pd.concat([df, new_col], axis = 1)
        
    return df

def _sector_rel_return(df, time_horiz):
    """HELPER to create sector_rel return Feature and add it to the surprise df."""
    sectors = df['factset_sector_num']
    col_name = time_horiz+'_rtn_qtr_end'
    rtns = df[col_name]
    temp = pd.concat([sectors, rtns], axis=1)
    new_col = temp.groupby('factset_sector_num').transform(lambda x: x - x.mean())
    new_col.rename(columns={col_name: time_horiz+'_sec_rel_rtn_F'}, inplace=True)
    df = pd.concat([df, new_col], axis = 1)
    
    return df



def _clean_feature_bind(surp_df, feature_df, retained_columns):
    """HELPER to create a tidied up df from a surp_df and feature_df, based on an Index object of cols to retain."""
    
    bound_df = pd.merge(surp_df, feature_df[retained_columns], on='unique_earnings_code')
    
    return bound_df


def _tidyfy_surp_df(df):
    """HELPER to clean up Surp dataframes prior to join"""
    
    _transform_dates(df, 't-7')
        
    new_cols = ['ticker_symbol',
                 'co_name',
                 'unique_earnings_code',
                 'factset_sector_num',
                 'factset_ind_num',
                 'calendar_qtr',
                 'fiscal_qtr',
                 'adtv_prev_month',
                 'report_date',
                 't-7',
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
    tidy_df = tidy_df.drop_duplicates()
    
    return tidy_df

def _transform_dates(df, col_name):
    """HELPER to convert date columns to a sortable format"""
    dates = list(df[str(col_name)])
    new_dates = []

    for d in dates:
        dt = datetime.strptime(d, '%m/%d/%Y')
        reformatted = str(dt)
        reformatted = reformatted[:10]
        new_dates.append(reformatted)
        
    df[str(col_name)] = new_dates

def _oos_partition(filename):
    """HELPER to create out-of-sample partition of 3Q18 data."""
    
    # read file in as a df and drop unwanted column
    data = pd.read_csv('data/'+str(filename)+'.csv', low_memory=False)
    data.drop(columns='Unnamed: 0', inplace=True)
    
    # reformat dates and sort by dates, ascending
    _transform_dates(data, 'report_date')
    
    data.sort_values(by=['report_date'], inplace=True)
        
    # set index to unique_earnings_code
    data.set_index('unique_earnings_code', inplace=True)
    
    # partition 3q18 data to test set
    test_partition = data[data.index.str.endswith('3Q18')]
    test_partition.to_csv('data/oos_data_partition.csv')
    
    # create y_oos array
    y_oos = test_partition.targets.values
    
    # create X_oos array
    features = test_partition.columns.str.endswith('F')
    X_oos = test_partition.values[:,features]
    
    # remove test_partition from data df
    data = data[data.index.str.endswith('3Q18') == False]
    
    # create y_train and X_train arrays
    y = data.targets.values              
    r = data['rel_t+3_rtn']
    X = data.values[:,features]
    
     
    return X, X_oos, y, y_oos, r


def _reformat_qtrs(x):
    """HELPER to reformat string dates into a string format that is sortable chronologically"""
    qtr = x[:2]
    yr = x[2:]
    new_qtr = yr+'-'+qtr
    
    return new_qtr

def _create_hard_classes(prob_array, threshold):
    """HELPER to create hard classifications based on a probability array and a given threshold"""
    hard_classes = []
    for row in prob_array:
        if row[1] >= threshold:
            hard_classes.append(1)
        else:
            hard_classes.append(0)
    return hard_classes


###################################################################################################

# script functions

def write_merged_frames(surp_lst, features_lst):
    """Create combined dataframes from two lists of dataframe names: surp & features"""
    
    combined_df_lst = []
    
    for s_df, f_df in zip(surp_lst, features_lst):
        #create quarter tag
        tag = s_df[-8:]

        # read surp df
        surp_df = pd.read_csv('data/'+s_df)
        surp_df = surp_df.dropna(subset=['factset_sector_num'])
        
        tidy_surp_df = _tidyfy_surp_df(surp_df)
        
        # read feature df
        feature_df = pd.read_csv('data/'+f_df)
        feature_df = feature_df.dropna(subset=['factset_sector_num'])

        # add calculated features
        feature_df = _sector_rel_earn_yld(feature_df)
        feature_df = _rel_earn_yld(feature_df)
        feature_df = _sector_rel_return(feature_df, '1y')
        feature_df = _sector_rel_return(feature_df, '6m')
        feature_df = _sector_rel_return(feature_df, '3m')

        # create list of columns to retain
        retained_cols = _create_feature_df_index(feature_df)

        # create combined df
        combined_df = _clean_feature_bind(tidy_surp_df, feature_df, retained_cols)
        

        # write combined_df to a csv file and store in data folder
        combined_df.to_csv('data/combined_'+tag, index=False)
        
        # record df written to a list
        combined_df_lst.append('combined_'+tag)
    
    return combined_df_lst


def stack_frames(sequence):
    """Concatenate combined dataframes and perform some basic data cleaning
    OUTPUT: writes combined_full_set.csv to data folder
    """
    sequence = sequence.copy()
    
    # concatenate dataframes
    combined_full = pd.concat([pd.read_csv(f'data/{file}', low_memory=False) for file in sequence])
    
    # drop empty rows
    combined_full = combined_full.drop_duplicates()
    
    # convert excel error codes to NaNs
    combined_full['rtn_t+3'] = combined_full['rtn_t+3'].apply(pd.to_numeric, args=('coerce',))
    combined_full['est_spread'] = combined_full['est_spread'].apply(pd.to_numeric, args=('coerce',))
    
    # drop rows with missing values in ticker_symbol column and NaNs in rtn_t+3 column
    combined_full = combined_full.dropna(subset=['ticker_symbol', 'rtn_t+3'])
    
    # add very small number to zero values in est_spread column to avoid division by zero
    combined_full['est_spread'] = combined_full['est_spread'].apply(pd.to_numeric, args=('coerce',))
    combined_full['est_spread'] = combined_full['est_spread'].apply(lambda x: x+0.0025 if x==0 else x)
    
    combined_full['spread_adj_surp'] = combined_full['spread_adj_surp'].apply(pd.to_numeric, args=('coerce',))
    combined_full = combined_full.dropna(subset=['spread_adj_surp'])
    
    combined_full.to_csv('data/combined_full_set.csv')
    
    return 


def create_labels(filename):
    """Creates target labels for the data set.
    INPUT: csv filename, as text
    OUTPUT: Creates primary labels based on threshold values of +/- 5% for the rel_t+3_rtn and a
    minimum spread_adj_surp of +/- 0.25 and adds them to the existing data set, writing
    them into the csv provided in column named 'targets'. Also creates alernative labels based on 
    thresholds of +/- 7.5% for the rel_t+3_rtn and a minimum spread_adj_surp of +/- 0.50 and writes
    them into column named 'extreme_targets'.    
    """
    
    # empty container for labels
    labels = []
    
    # load data from file
    data = pd.read_csv('data/'+str(filename)+'.csv', low_memory=False)
    
    # pull out key columns and convert to lists
    event_rtn = list(data['rel_t+3_rtn'])
    event_react = list(data['spread_adj_surp'])
    
       
    # create labels based on conditions
    labels = []

    for rtn, react in (zip(event_rtn, event_react)):
        if (rtn <= -5) and (react >= 0.25):
            labels.append(1)
        #elif (rtn >= 5) and (react <= -0.25):
            #labels.append(2)
        else:
            labels.append(0)
            
    # create alternative extreme targets
    xtrm_tgts = []
    
    for rtn, react in (zip(event_rtn, event_react)):
        if (rtn <= -7.5) and (react >= 0.50):
            xtrm_tgts.append(1)
        #elif (rtn >= 7.5) and (react <= -0.50):
            #xtrm_tgts.append(2)
        else:
            xtrm_tgts.append(0)
       
    # add classification targets to combined dataframe
    data.insert(loc=21, column='targets', value=labels)
    data.insert(loc=22, column='extreme_targets', value=xtrm_tgts)
    
    # drop Unnamed column
    data.drop(columns='Unnamed: 0', inplace=True)
     
    # overwrite passed file with updated df
    data.to_csv('data/'+str(filename)+'.csv')
    
    
    return


def clean_features(filename):
    """Cleans features of filename passed to function as a string.
       * Eliminates rows with excessive missing values.
       * Fills in individual missing values in columns based on industry group average for quarter.
       * Removes rows for stocks with fewer than 4 analyst estimates for a given quarter.
       OUTPUT: Writes combined_clean.csv to data folder.
    """
    
    
    # load data from file
    data = pd.read_csv('data/'+str(filename)+'.csv', low_memory=False)
    
    # eliminate observations with excessive missing values
    bad_rows = data.iloc[:,22:].isnull().sum(axis=1) > 4
    data = data[bad_rows == False]
    
    # replace missing column values with qtr averages for industry groups
    temp = data.groupby(['calendar_qtr','factset_ind_num']).transform(lambda data: data.fillna(data.mean()))

    for col in temp:
        data[col] = temp[col]
        
    # scrub any remaining missing values by dropping observations
    bad_rows = data.iloc[:,22:].isnull().sum(axis=1) > 0
    data = data[bad_rows == False]
    
    # convert adtv column to numeric
    data['adtv_prev_month'] = data['adtv_prev_month'].apply(pd.to_numeric, args=('coerce',))

    # remove observations with fewer than 4 analyst estimates for the quarter
    data = data[(data['num_ests_qtr_end'] > 3) & (data['adtv_prev_month'] >= 15)]
    
    
    # drop Unnamed column
    data.drop(columns='Unnamed: 0', inplace=True)    
    
    # write output to a new file
    data.to_csv('data/combined_clean.csv')    
        
    return

    
def encode_sectors(filename):
    """Encodes factset_sector_num as binary categorical columns and writes them to the csv file
    with 'trailing F' notation."""
    data = pd.read_csv('data/'+str(filename)+'.csv', low_memory=False)
    data.drop(columns='Unnamed: 0', inplace=True)
    
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    sectors = data.factset_sector_num.values
    sectors = sectors.reshape(-1, 1)
    encoded_cols = enc.fit_transform(sectors)
    feature_labels = list(enc.get_feature_names())
    feature_names = []
    
    for item in feature_labels:
        tagged_name = item+'_F'
        feature_names.append(tagged_name)
    
    temp = pd.DataFrame(encoded_cols, columns=feature_names)
    data = pd.concat([data, temp], axis=1)
    # write combined_df to a csv file and store in data folder
    data.to_csv('data/'+str(filename)+'.csv')
    
    return

def add_vol_data():
    data = pd.read_csv('data/combined_clean.csv', low_memory=False)
    data.drop(columns='Unnamed: 0', inplace=True)
    vol_data = pd.read_csv('data/vol_data_cleaned.csv', low_memory=False)
    merged_data = pd.merge(data, vol_data, on='unique_earnings_code')
    merged_data['h30_v_imp30_F'] = merged_data['h30_v_imp30_F'].apply(lambda x: np.nan_to_num(x))
    merged_data.groupby(['calendar_qtr','factset_ind_num']).transform(lambda data: data.fillna(data.mean()))
    merged_data.to_csv('data/combined_clean.csv')




def prepare_partitions(filename, test_slice=0.25): # removed rand_seed
    """Partitions 3Q18 data for out of sample validation and creates train-test split on remainder. 
            * Writes 3Q18 df to a file in the data folder: oos_data_partition.csv
            * Creates a train-test split on remaining data
            * Returns X_train, X_test, X_oos, y_train, y_test, y_oos
            
        test_slice defaults to 0.25, but can be assigned by keyword
        rand_seed defaults to 1970, but can be assigned by keyword
    """
    
    # remove final out-of-sample data (3q18)
    X, X_oos, y, y_oos, r = _oos_partition(filename)

    # create test_train split on full data set, less 3Q18 out-of-sample data
    X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(X,y, r, test_size=test_slice)
     
    return X_train, X_test, y_train, y_test, r_train, r_test



# end pre-processing & data partitioning section
###################################################################################################

# model fitting & prediction

def rf_analysis(filename, trees, features_per_split, crossval_folds=5):
    """docstring"""
    
    
    
    # partition dataset and extract train/test split
    X_train, X_test, y_train, y_test, r_train, r_test = prepare_partitions(filename)
        
    # make sure float type is correct
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
     
    # instantiate model
    clf = RandomForestClassifier(n_estimators = trees,
                                criterion = 'gini',
                                max_features = features_per_split)
    
    # fit model
    clf.fit(X_train, y_train)
    
    # set up cross validation
    skf = StratifiedKFold(n_splits=crossval_folds, shuffle=True)
    
    # generate log loss from cross validation
    cv_log_loss = cross_val_score(clf, # model
                             X_train, # Feature matrix
                             y_train, # Target vector
                             cv=skf, # Cross-validation technique
                             scoring='neg_log_loss', # Loss function
                             n_jobs=-1) # Use all CPU scores
    
      
    # calculate average cross-validated log loss
    avg_log_loss = np.mean(cv_log_loss) * -1
    
   
    # generate probability predictions
    y_predict = clf.predict_proba(X_test)
        
    # create rf_output dict
    rf_output = {'avg_log_loss' : avg_log_loss,
                       'cv_log_loss' : list(cv_log_loss),
                       'y_test' : list(y_test),
                       'y_predict' : list(y_predict),
                     'rtns_test' : list(r_test.values)}
                             
    


    
    return rf_output
    

def calc_profit_curve(y_test, y_predict, max_threshold):
    thresholds = []
    results = []
    for i in range(5, max_threshold+1):
        thresh = float(i/100)
        hard_classes = _create_hard_classes(y_predict, thresh)
        tn, fp, fn, tp = confusion_matrix(y_test, hard_classes).ravel()
        profit = (5*tp*0.10)+(5*fp*-0.01)
        thresholds.append(i)
        ppv = tp / (tp+fp)
        result_entry = [tp, fp, ppv, profit]
        results.append(result_entry)
        
    return dict(zip(thresholds, results))    

def simulate_perf(y_test, y_predict, r_test, max_threshold, trade_size=5, min_threshold=5):
    rtns = r_test
    
    output = {}
    total_output = {}
    
    for i in range(min_threshold, max_threshold+1):
        thresh = float(i/100)
        hard_classes = _create_hard_classes(y_predict, thresh)
        
        for label, rtn in zip(hard_classes, rtns):
            if label == 1:
                
                profit = trade_size * -rtn/100
                
                
                if thresh in output.keys():
                    output[thresh].append(profit)
                else:
                    output[thresh] = [profit]
    
    for key, val in output.items():
        total_output[key] = np.sum(val)
    
        
    return total_output

def simulate_strat(y_predict, r_test, trade_thresh, upsize_thresh, trade_size=5, upsize_factor=2):
    target_probs = []
    trades = []
    
    for item in y_predict:
        target_probs.append(item[1])
    
    
        
    for prob, rtn in zip(target_probs, r_test):
        if (prob >= trade_thresh) & (prob < upsize_thresh):
                
                profit = trade_size * -rtn/100
                trades.append(profit)
                                
        elif prob >= upsize_thresh:        
                
                profit = (trade_size*upsize_factor) * -rtn/100
                trades.append(profit)
    
    return sum(trades), trades


















# end model fitting & prediction section
###################################################################################################

# plotting

def plot_returns_hist(filename):    
    sns.set()
    df = pd.read_csv('data/'+filename+'.csv', low_memory=False)
    df.drop(columns='Unnamed: 0', inplace=True)
    returns = df['rel_t+3_rtn']
    fig, ax = plt.subplots(figsize= (8, 5))
    plt.tight_layout()
    ax = sns.distplot(returns, kde=False, bins=256)
    _ = ax.set_title("Earnings Event Return Distribution")
    _ = ax.set_xlabel("Relative Returns: t-1 to t+3")
    _ = ax.set_ylabel("Proportion of Data")
    _ = ax.set_xlim(left=-50, right=50)
    _ = plt.subplots_adjust(bottom=0.12)
    plt.savefig('viz/rtns_all.png')



def obs_by_qtr_bars(filename):
    sns.set()
    df = pd.read_csv('data/'+filename+'.csv', low_memory=False)
    df.drop(columns='Unnamed: 0', inplace=True)   
    df['calendar_qtr'] = df['calendar_qtr'].apply(_reformat_qtrs)
    obs_by_qtr = df.groupby(['calendar_qtr'])['targets'].count()
    fig, ax = plt.subplots()
    x = np.arange(len(obs_by_qtr.values))
    bars = ax.bar(x, obs_by_qtr.values)
    _ = ax.set_xticks(x)
    _ = ax.set_xticklabels(obs_by_qtr.index, rotation=90)
    _ = ax.set_ylim(bottom=1100, top=1400)
    _ = ax.set_title("Observations by Calendar Quarter")
    _ = ax.set_xlabel("Calendar Quarters")
    _ = ax.set_ylabel("Number of Observations")
    _ = plt.subplots_adjust(bottom=0.2)
    plt.savefig('viz/obs_by_qtr.png')


def targets_df(filename):
    df = pd.read_csv('data/'+filename+'.csv', low_memory=False)
    df.drop(columns='Unnamed: 0', inplace=True)
    df['calendar_qtr'] = df['calendar_qtr'].apply(_reformat_qtrs)
    obs_by_qtr = df.groupby(['calendar_qtr'])['targets'].count()
    labels_by_qtr = df.groupby(['calendar_qtr', 'targets'])['ticker_symbol'].count()
    targets_count = pd.DataFrame(labels_by_qtr).reset_index('targets')
    targets_count = targets_count[targets_count['targets'] == 1]
    targets_count = targets_count.drop(['targets'], axis=1)
    targets_count['observations'] = obs_by_qtr.values
    targets_count.rename(columns={'ticker_symbol':'targets'}, inplace=True)
    targets_count['target_pct'] = (targets_count['targets'] / targets_count['observations'])*100
    
    targets_count.to_csv('data/targets_df.csv')
    return targets_count


def targets_pct_plot(filename):
    df = pd.read_csv('data/'+filename+'.csv', low_memory=False)    
    df.reset_index(inplace=True)
    
    sns.set()
    fig, ax = plt.subplots()
    x = np.arange(len(df['target_pct'].values))
    bars = ax.bar(x, df['target_pct'].values)
    _ = ax.set_xticks(x)
    _ = ax.set_xticklabels(df['calendar_qtr'], rotation=90)
    _ = ax.set_ylim(bottom=3, top=11)
    _ = ax.set_title("Targets: Percent of Observations by Quarter")
    _ = ax.set_xlabel("Calendar Quarters")
    _ = ax.set_ylabel("Pct of Observations")
    _ = plt.subplots_adjust(bottom=0.22)
    plt.savefig('viz/targets_pct_plot.png')










# end plotting section
###################################################################################################

# sequential simulation

def create_sim_data_sets():
    """Creates training and test data sets for running sequential simulations based on eight
    quarters of training data."""
    
    # list of quarters of data available
    data_qtrs = ['1Q14',
             '2Q14', 
             '3Q14',
             '4Q14', 
             '1Q15',
             '2Q15',
             '3Q15',
             '4Q15',
             '1Q16',
             '2Q16',
             '3Q16',
             '4Q16',
             '1Q17',
             '2Q17',
             '3Q17',
             '4Q17',
             '1Q18',
             '2Q18',
             '3Q18']
    
    # open combined_clean data
    D = pd.read_csv('data/combined_clean.csv', low_memory=False)
    D.drop(columns='Unnamed: 0', inplace=True)
    
    
    # sliding window counter
    start_window = 0
    end_window = 8
    
    # loop through data
    for i in range(1, 12):
        
        # create train sets
        train_qtrs = data_qtrs[start_window:end_window]
        train_temp = D[D['calendar_qtr'].isin(train_qtrs)]
        train_temp.to_csv('data/sim_training_sets/training_'+data_qtrs[end_window]+'.csv')
        
        
        # create test sets
        test_qtr = data_qtrs[end_window]
        test_temp = D[D['calendar_qtr'] == test_qtr]
        test_temp.to_csv('data/sim_test_sets/test_'+data_qtrs[end_window]+'.csv')
        
        # step counters       
        start_window += 1
        end_window += 1


def sequential_sim_analysis(qtrs_list, trees=1000, features_per_split=4):
    """Pass a list of qtrs (ranging from 1Q16 through 3Q16) to the function to generate model outputs."""
    
    counter = 0
    
    for i in range(0, len(qtrs_list)):
        # load training data
        train_data = pd.read_csv('data/sim_training_sets/training_'+qtrs_list[counter]+'.csv',
                                low_memory=False)
        
        # load target data
        target_data = pd.read_csv('data/sim_test_sets/test_'+qtrs_list[counter]+'.csv',
                                low_memory=False)
    
    
        # create features list
        features = train_data.columns.str.endswith('F')
    
        # create y_train and X_train arrays
        y_train = train_data.targets.values              
        X_train = train_data.values[:,features]
        
        # create y_test and X_test arrays
        y_target = [int(v) for v in target_data.targets.values]         
        X_target = target_data.values[:,features]
        
        # make sure numeric types are correct
        X_train = X_train.astype(float)
        X_target = X_target.astype(float)
                
        # create return array
        rtns_target = target_data['rel_t+3_rtn']
        
        # create ticker list
        tickers = target_data['ticker_symbol']
        
        
        # instantiate model
        clf = RandomForestClassifier(n_estimators = trees,
                                criterion = 'gini',
                                max_features = features_per_split)
        
        
        # fit model
        clf.fit(X_train, y_train)
        
        
        # set up cross validation
        skf = StratifiedKFold(n_splits=5, shuffle=True)
    
        # generate log loss from cross validation
        cv_log_loss = cross_val_score(clf, # model
                             X_train, # Feature matrix
                             y_train, # Target vector
                             cv=skf, # Cross-validation technique
                             scoring='neg_log_loss', # Loss function
                             n_jobs=-1) # Use all CPU scores
    
      
        # calculate average cross-validated log loss
        avg_log_loss = np.mean(cv_log_loss) * -1
    
   
        # generate probability predictions
        y_predict = clf.predict_proba(X_target)
        y_predict = y_predict.tolist()
        
        
             

        # create output dictionary
        output = {'target_qtr': qtrs_list[counter],
                    'avg_log_loss' : avg_log_loss,
                       'cv_log_loss' : list(cv_log_loss),
                  'ticker_symbols' : list(tickers.values),
                       'y_test' : list(y_target),
                       'y_predict' : list(y_predict),
                     'rtns_test' : list(rtns_target.values)}
        
        
        
        # dump output dictionaries to json files
        with open('data/sim_output/results_'+qtrs_list[counter]+'.json', 'w') as fp:
            json.dump(output, fp)
        
        # check length of lists
        print(len(output['ticker_symbols']))
        print(len(output['y_test']))
        print(len(output['y_predict']))
        print(len(output['rtns_test']))
        
        
        
        counter += 1