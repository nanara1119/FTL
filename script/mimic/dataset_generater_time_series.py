# %%
import pandas as pd
import os
from helpers import *
from sklearn.impute import SimpleImputer

# %%
print(os.getcwd())
master_dataset_path = "../../clean_data/"
output_path = "../../clean_data/"


df_master = pd.read_csv(os.path.join(master_dataset_path, "master_dataset.csv"))

pd.set_option('display.max_columns', 200)
df_master.head()

# %%
print('Before filtering for "age" >= 18 : master dataset size = ', len(df_master))
df_master = df_master[df_master['age'] >= 18]
print('After filtering for "age" >= 18 : master dataset size = ', len(df_master))

print('Before filtering for non-null "triage_acuity" >= 18 : master dataset size = ', len(df_master))
df_master = df_master[df_master['triage_acuity'].notnull()]
print('After filtering for non-null "triage_acuity" >= 18 : master dataset size = ', len(df_master))

# %%

# from mimic-extract
vitals_valid_range = {
    'temperature': {'outlier_low': 14.2, 'valid_low': 26, 'valid_high': 45, 'outlier_high':47},
    'heartrate': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 350, 'outlier_high':390},
    'resprate': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 300, 'outlier_high':330},
    'o2sat': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 100, 'outlier_high':150},
    'sbp': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 375, 'outlier_high':375},
    'dbp': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 375, 'outlier_high':375},
    'pain': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 10, 'outlier_high':10},
    'acuity': {'outlier_low': 1, 'valid_low': 1, 'valid_high': 5, 'outlier_high':5},
}

df_master = convert_temp_to_celcius(df_master)

display_outliers_count(df_master, vitals_valid_range)

# %%
df_master = remove_outliers(df_master, vitals_valid_range)

# %%
df_train=df_master.sample(frac=0.8,random_state=10) #random state is a seed value
df_test=df_master.drop(df_train.index)

print('Training dataset size = ', len(df_train))
print('Testing dataset size = ', len(df_test))

df_train.head()

# %%
df_missing_stats = df_train.isnull().sum().to_frame().T
df_missing_stats.loc[1] = df_missing_stats.loc[0] / len(df_train)
df_missing_stats.index = ['no. of missing values', 'percentage of missing values']
df_missing_stats

vitals_cols = [col for col in df_master.columns if len(col.split('_')) > 1 and 
                                                   col.split('_')[1] in vitals_valid_range and
                                                   col.split('_')[1] != 'acuity']
vitals_cols

imputer = SimpleImputer(strategy='median')
df_train[vitals_cols] = imputer.fit_transform(df_train[vitals_cols])
df_test[vitals_cols] = imputer.transform(df_test[vitals_cols])

# %%
add_triage_MAP(df_test) # add an extra variable MAP
add_score_CCI(df_test)
add_score_CART(df_test)
add_score_REMS(df_test)
add_score_NEWS(df_test)
add_score_NEWS2(df_test)
add_score_MEWS(df_test)

add_triage_MAP(df_train) # add an extra variable MAP
add_score_CCI(df_train)
add_score_CART(df_train)
add_score_REMS(df_train)
add_score_NEWS(df_train)
add_score_NEWS2(df_train)
add_score_MEWS(df_train)

# %%
df_train.to_csv(os.path.join(output_path, 'train.csv'), index=False)
df_test.to_csv(os.path.join(output_path, 'test.csv'), index=False)

# %%
raw_data_path = '../../raw_data/'
df_vitalsign = read_vitalsign_table(os.path.join(raw_data_path, 'ed/vitalsign.csv'))

df_vitalsign['charttime'] = pd.to_datetime(df_vitalsign['charttime'])
df_vitalsign.sort_values('charttime', inplace=True)
df_vitalsign.drop('ed_rhythm', axis=1, inplace=True)
df_vitalsign.head()

# %%
grouped = df_vitalsign.groupby('stay_id')

resample_freq = '1H' # 1 hour
# resample_freq = '30T' # 30 minutes

df_list = []
counter = 0
N = len(grouped)
for stay_id, stay_df in grouped:
    counter += 1
    stay_df.set_index('charttime', inplace=True)
    stay_df = stay_df.resample('1T', origin='start').interpolate(method='linear')\
                     .resample(resample_freq, origin='start').asfreq().ffill().bfill()
    if counter % 10000 == 0 or counter == N:
        print('%d/%d' % (counter, N), end='\r')
    df_list.append(stay_df)

# %%
df_vitalsign_resampled = pd.concat(df_list)

df_vitalsign_resampled = convert_temp_to_celcius(df_vitalsign_resampled)

df_vitalsign_resampled = remove_outliers(df_vitalsign_resampled, vitals_valid_range)

vitals_cols = [col for col in df_vitalsign_resampled.columns if len(col.split('_')) > 1 and 
                                                                col.split('_')[1] in vitals_valid_range]
print(vitals_cols)

imputer = SimpleImputer(strategy='median')
df_vitalsign_resampled[vitals_cols] = imputer.fit_transform(df_vitalsign_resampled[vitals_cols])

df_vitalsign_resampled.to_csv(os.path.join(output_path, 'ed_vitalsign_'+resample_freq+'_resampled.csv'))

