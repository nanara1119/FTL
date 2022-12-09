# %%
import os
import time
import random
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import optimizers, metrics
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from mimic.helpers import PlotROCCurve
from performance import output

# %%
print(os.getcwd())
data_path = "../clean_data"
path = "../result/1_hospitalization"
output_path = os.path.join(path, "Figure3")
df_train = pd.read_csv((os.path.join(data_path, 'train.csv')))
df_test = pd.read_csv((os.path.join(data_path, 'test.csv')))
confidence_interval = 95
random_seed=9

random.seed(random_seed)
np.random.seed(random_seed)

# %%
pd.set_option('display.max_columns', 100) 
pd.set_option('display.max_rows', 100) 
df_train.head()

print('training size =', len(df_train), ', testing size =', len(df_test))

# %%
variable = ["age", "gender", 
            
            "n_ed_30d", "n_ed_90d", "n_ed_365d", "n_hosp_30d", "n_hosp_90d", 
            "n_hosp_365d", "n_icu_30d", "n_icu_90d", "n_icu_365d", 
            
            "triage_temperature", "triage_heartrate", "triage_resprate", 
            "triage_o2sat", "triage_sbp", "triage_dbp", "triage_pain", "triage_acuity",
            
            "chiefcom_chest_pain", "chiefcom_abdominal_pain", "chiefcom_headache",
            "chiefcom_shortness_of_breath", "chiefcom_back_pain", "chiefcom_cough", 
            "chiefcom_nausea_vomiting", "chiefcom_fever_chills", "chiefcom_syncope", 
            "chiefcom_dizziness", 
            
            "cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia", 
            "cci_Pulmonary", "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1", 
            "cci_DM2", "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2", 
            "cci_Cancer2", "cci_HIV", 
            
            "eci_Arrhythmia", "eci_Valvular", "eci_PHTN",  "eci_HTN1", "eci_HTN2", 
            "eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy", 
            "eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss",
            "eci_Anemia", "eci_Alcohol", "eci_Drugs","eci_Psychoses", "eci_Depression"]

outcome = "outcome_hospitalization"

# %%
X_train = df_train[variable].copy()
y_train = df_train[outcome].copy()
X_test = df_test[variable].copy()
y_test = df_test[outcome].copy()

X_train.dtypes.to_frame().T

# %%
encoder = LabelEncoder()
X_train['gender'] = encoder.fit_transform(X_train['gender'])
X_test['gender'] = encoder.transform(X_test['gender'])

# %%
print('class ratio')
ratio = y_train.sum()/(~y_train).sum()
print('positive : negative =', ratio, ': 1')


# %%
result_list = []

def build_model():
    model = tf.keras.Sequential([
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss="binary_crossentropy", 
        optimizer=optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy', 'AUC', {'auprc': metrics.AUC(name='auprc', curve='PR')}, 
                       'TruePositives', 'TrueNegatives', 'Precision', 'Recall'])
    
    return model

# %%
model = build_model()
start_time = time.time()
model.fit(X_train.astype(np.float32), y_train, batch_size=200, epochs=20)
runtime = time.time() - start_time

# %%
probs = model.predict(X_test.astype(np.float32))
result = PlotROCCurve(probs,y_test, ci=confidence_interval, random_seed=random_seed)
results = ["MLP"]
results.extend(result)
results.append(runtime)
result_list.append(results)

# %%
print(result_list)
output(result_list, path, "1_hospitalization_triage_2") 