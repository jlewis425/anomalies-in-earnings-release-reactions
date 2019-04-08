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

# create list of 'surprise' files

surp_files = (['surp_1q14.csv',
               'surp_2q14.csv',
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
               'surp_2q18.csv',
               'surp_3q18.csv'])

# create list of 'features' files

features_files = (['features_1q14.csv',
                   'features_2q14.csv',
                   'features_3q14.csv',
                   'features_4q14.csv',
                   'features_1q15.csv',
                   'features_2q15.csv',
                   'features_3q15.csv',
                   'features_4q15.csv',
                   'features_1q16.csv',
                   'features_2q16.csv',
                   'features_3q16.csv',
                   'features_4q16.csv',
                   'features_1q17.csv',
                   'features_2q17.csv',
                   'features_3q17.csv',
                   'features_4q17.csv',
                   'features_1q18.csv',
                   'features_2q18.csv',
                   'features_3q18.csv'])


# pipeline script

combined_frames = write_merged_frames(surp_files, features_files)
combined_full = stack_frames(combined_frames)
create_labels('combined_full_set')
clean_features('combined_full_set')


# partition data
X, X_oos, y, y_oos, features = partition_dataset('combined_clean') # create oos partition
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1970) #create train-test split

# random forest model
clf = RandomForestClassifier(n_estimators=500, 
                             criterion='gini',
                             max_features=4,
                             max_depth=6,
                             random_state=1970)

clf.fit(X_train, y_train)

# set kfold cross-val
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1970)

# predict probabilities
y_pred = clf.predict_proba(X_test)

