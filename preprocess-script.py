# %% [markdown]
# # Settings

# %%


# %% [markdown]
# # Dataset Overview

# %% [markdown]
# ## Partcipants

# %%
import os
from pathlib import Path

import pandas as pd


from Funcs.Utility import  *

PARTICIPANTS = pd.read_csv(PATH_PARTICIPANT).set_index('pcode')

PARTICIPANTS.to_csv(os.path.join(PATH_INTERMEDIATE, 'proc', 'PARTICIPANT_INFO.csv'),index = True)

rri_existing = set((p.stem[0:3] for p in Path('Intermediate/proc/').glob('*RRI*.csv')))
hrt_existing = set((p.stem[0:3] for p in Path('Intermediate/proc/').glob('*HRT*.csv')))

rri_missing = rri_existing.difference((f'{i:02}' for i in range(1,81) if i not in (22,27,59,65)))
hrt_missing = hrt_existing.difference((f'{i:02}' for i in range(1,81) if i not in (22,27,59,65)))


hri_rri_missing = rri_missing.union(rri_existing)

PARTICIPANTS = PARTICIPANTS.loc[list(hri_rri_missing),:]

# %% [markdown]
# ## Labels (via ESM)

# %%
import pandas as pd
import os

LABELS = pd.read_csv(PATH_ESM).assign(
    timestamp=lambda x: pd.to_datetime(x['responseTime'], unit='ms', utc=True).dt.tz_convert(DEFAULT_TZ)
).set_index(
    ['pcode', 'timestamp']
)


# %% [markdown]
# # Preprocessing

# %% [markdown]
# ## Labels

# %%
# LABELS_VALID = LABELS.loc[
#     lambda x: ~x['scheduledTime'].isna(), :
# ]
# print(f'# Non-voluntary response: {len(LABELS_VALID)}')
# print(summary(LABELS_VALID.groupby('pcode').count().iloc[:, -1]))
# LABELS_VALID = LABELS

# excl_pcode = LABELS_VALID.loc[
#     lambda x: ~x['scheduledTime'].isna()
# ].groupby('pcode').count().iloc[:, -1].loc[lambda y: y < 35]

LABELS_VALID = LABELS

excl_pcode = LABELS_VALID.groupby('pcode').count().iloc[:, -1].loc[lambda y: y < 35]

# excl_pcode = LABELS_VALID.loc[
#     lambda x: ~x['scheduledTime'].isna()
# ].groupby('pcode').count().iloc[:, -1].loc[lambda y: y < 35]

LABELS_VALID = LABELS_VALID.loc[
    lambda x:  ~x.index.get_level_values('pcode').isin(excl_pcode.index), :
]
print(f'# Response from participants with enough responses: {len(LABELS_VALID)}')
print(summary(LABELS_VALID.groupby('pcode').count().iloc[:, -1]))

print('# Participants whose responses to ESM delivery were less then 35')
print(excl_pcode, f'#participants = {len(excl_pcode)} / #response = {sum(excl_pcode)}')

# %%
#Drop duplicate responses
LABELS_VALID = LABELS_VALID.groupby('pcode').apply(lambda x: x.reset_index(drop=False).drop_duplicates(subset='timestamp', keep='first')).set_index(
    ['pcode', 'timestamp']
)

# %%
import pandas as pd
import numpy as np

conditions = [
    (LABELS_VALID['stress'] < 0), 
    (LABELS_VALID['stress'] == 0), 
    (LABELS_VALID['stress'] > 0)
]

choices = [0, 1, 2]  # correspondingly negative, zero and positive

LABELS_PROC = LABELS_VALID.assign(
    valence_fixed = lambda x: np.where(x['valence'] > 0, 1, 0),
    arousal_fixed = lambda x: np.where(x['arousal'] > 0, 1, 0),
    stress_fixed = lambda x: np.where(x['stress'] > 0, 1, 0),
    disturbance_fixed = lambda x: np.where(x['disturbance'] > 0, 1, 0),   
    stress_fixed_tri = np.select(conditions, choices, default=np.nan),

)
LABELS_PROC.head()

# %%
import numpy as np

def zscore(col):
    mean = col.mean()
    std = col.std()
    return (col - mean) / std

# Calculate the overall mean z-score
LABELS_PROC['zscore'] = LABELS_PROC['stress'].transform(zscore)
overall_mean_zscore = LABELS_PROC['zscore'].mean()

# Binarize using the overall mean z-score
LABELS_PROC['stress_user_mean'] = (LABELS_PROC['zscore'] > overall_mean_zscore).astype(int)

# %%
LABELS_PROC.to_csv(os.path.join(PATH_INTERMEDIATE, 'proc', 'LABELS_PROC.csv'), index=True)

# %% [markdown]
# ## Sensor Data

# %%
import pandas as pd
import scipy.spatial.distance as dist
from typing import Dict, Union
import pygeohash as geo
from datetime import timedelta
from collections import defaultdict  
from scipy.signal import medfilt
from sklearn.preprocessing import MinMaxScaler

def trim_outlier(col, threshold=3.0):
    """
    Remove the values in a dataframe column based on the median and the median absolute deviation.

    Parameters
    ----------
    col : pandas.Series
        The column to be trimmed.
    threshold : float, optional
        The threshold for trimming, expressed in units of the Median Absolute Deviation (MAD).
        Observations with a distance greater than `threshold` times the MAD value from the median are removed.
        Default is 3.0.

    Returns
    -------
    pandas.Series
        The column without outliers.
    """
    median = col.median()
    mad = (col - median).abs().median()
    threshold_value = threshold * mad
    mask = (col > median - threshold_value) & (col < median + threshold_value)
    return col[mask]

# %%
import pandas as pd
import numpy as np
import scipy.spatial.distance as dist
from typing import Dict, Union
#import pygeohash as geo
from sklearn.cluster import DBSCAN
from datetime import timedelta
from collections import defaultdict
from poi import PoiCluster
from Funcs.Utility import transform
import warnings
from pandas.errors import PerformanceWarning

warnings.simplefilter(action='ignore', category=PerformanceWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)



# AmbientLight.csv
def _proc_ambient_light(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    return data['brightness'].astype('float32')
    

# StepCount.csv
def _proc_step_count(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    new_data = []

    for pcode in data.index.get_level_values('pcode').unique():
        sub = data.loc[(pcode, ), :].sort_index(
            axis=0, level='timestamp'
        ).assign(
            steps=lambda x: (x['totalSteps'] - x['totalSteps'].shift(1)),
            pcode=pcode
        ).reset_index()
        new_data.append(sub)

    new_data = pd.concat(new_data, axis=0, ignore_index=True).set_index(
        ['pcode', 'timestamp']
    )

    return new_data['steps'].dropna().astype('float32')
    


# Acceleration.csv
def _proc_acceleration(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    data = data.assign(
        mag=lambda x: np.sqrt(np.square(x['x']) + np.square(x['y']) + np.square(x['z']))
    )

    return {
        'AXX': data['x'].astype('float32'),
        'AXY': data['y'].astype('float32'),
        'AXZ': data['z'].astype('float32'),
        'MAG': data['mag'].astype('float32')
    }

# SkinTemperature.csv
def _proc_skin_temperature(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    temperature = []
    for pcode in data.index.get_level_values('pcode').unique():
        v = data.loc[(pcode, ), :].sort_index(axis=0,level='timestamp').assign(pcode=pcode)
        v = v.reset_index()
        v['temperature'] = trim_outlier(v['temperature'], threshold=3.0)
        v= v[~v['temperature'].isnull()]
        # Z-score normalize column 'temperature'
        v['temperature'] = (v['temperature'] - v['temperature'].mean()) / v['temperature'].std()
        temperature.append(v)

    temperature = pd.concat(temperature, axis=0, ignore_index=True).set_index(
                ['pcode', 'timestamp']
            ) 
    
    return temperature['temperature'].astype('float32')


# RRI.csv
def _proc_rri(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    RRI = []
    for pcode in data.index.get_level_values('pcode').unique():
        v = data.loc[(pcode, ), :].sort_index(axis=0,level='timestamp').assign(pcode=pcode)
        v = v.reset_index()
        
        v= v[~v['interval'].isnull()]
        # Z-score normalize column 'interval'
        v['interval'] = (v['interval'] - v['interval'].mean()) / v['interval'].std()
        v['interval'] = trim_outlier(v['interval'], threshold=3.0)
        RRI.append(v)

    RRI = pd.concat(RRI, axis=0, ignore_index=True).set_index(
                ['pcode', 'timestamp']
            ) 
    return RRI['interval'].astype('float32')



# HR.csv
def _proc_hr(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    data['bpm'] = data.loc[(data['bpm'] >= 30) | (data['bpm'] <= 220), 'bpm']
    data= data[~data['bpm'].isnull()]
    HRT = []
    for pcode in data.index.get_level_values('pcode').unique():
        v = data.loc[(pcode, ), :].sort_index(axis=0,level='timestamp').assign(pcode=pcode)
        v = v.reset_index()
        
        v= v[~v['bpm'].isnull()]
        # Z-score normalize column 'bpm'
        v['bpm'] = (v['bpm'] - v['bpm'].mean()) / v['bpm'].std()
        v['bpm'] = trim_outlier(v['bpm'], threshold=3.0)
        HRT.append(v)

    HRT = pd.concat(HRT, axis=0, ignore_index=True).set_index(
                ['pcode', 'timestamp']
            ) 
    return HRT['bpm'].astype('float32')
    

# EDA.csv
def _proc_eda(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:

    # Apply a median filter with a window size of window_size_sec seconds
    window_size_sec = 5
    window_size = window_size_sec * 2  # Multiply by the sampling frequency (2 Hz)

   #Make the window size odd if it is even
    if window_size % 2 == 0:
        window_size += 1

    data["conductance"] = 1 / (data["resistance"] / 1000) # divide by 1000 to convert kΩ to Ω
    data['conductance'] =data.loc[(data['conductance'] >= 0.01) & (data['conductance'] <= 100), 'conductance']
    data= data[~data['conductance'].isnull()]


    eda = []
    for pcode in data.index.get_level_values('pcode').unique():
        v = data.loc[(pcode, ), :].sort_index(axis=0,level='timestamp').assign(pcode=pcode)
        v = v.reset_index()

        eda_data = v['conductance'].to_numpy()
        eda_data = medfilt(eda_data, window_size)
        # Reshape to 2D with a single column
        eda_data = eda_data.reshape(-1, 1)
#         eda_data = eda_data.reshape(-1)
        # assuming your data is a numpy array with shape (n_samples, n_features)
        scaler = MinMaxScaler()
        eda_data_scaled = scaler.fit_transform(eda_data)
        eda_data = scaler.inverse_transform(eda_data_scaled).reshape(-1)

        v['conductance'] =eda_data
        v= v[~v['conductance'].isnull()]

        eda.append(v)

    eda = pd.concat(eda, axis=0, ignore_index=True).set_index(
                ['pcode', 'timestamp']
            ) 
    
    return eda['conductance'].astype('float32')


# Distance.csv
def _proc_distance(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    new_data = []

    for pcode in data.index.get_level_values('pcode').unique():
        sub = data.loc[(pcode, ), :].sort_index(
            axis=0, level='timestamp'
        ).assign(
            distance=lambda x: x['totalDistance'] - x['totalDistance'].shift(1),
            pcode=pcode
        ).reset_index()

        new_data.append(sub)

    new_data = pd.concat(new_data, axis=0, ignore_index=True).set_index(
        ['pcode', 'timestamp']
    )

    return {
        'DST': new_data['distance'].dropna().astype('float32'),
        # 'MOT': new_data['motionType'].astype('object'),
        'PAC': new_data['pace'].astype('float32'),
        'SPD': new_data['speed'].astype('float32')
    }


# Calorie.csv
def _proc_calories(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    new_data = []

    for pcode in data.index.get_level_values('pcode').unique():
        sub = data.loc[(pcode, ), :].sort_index(
            axis=0, level='timestamp'
        ).assign(
            calories=lambda x: x['totalCalories'] - x['totalCalories'].shift(1),
            pcode=pcode
        ).reset_index()

        new_data.append(sub)

    new_data = pd.concat(new_data, axis=0, ignore_index=True).set_index(
        ['pcode', 'timestamp']
    )

    return new_data['calories'].dropna().astype('float32')    

# %%
import pandas as pd
import gc
from functools import reduce
import warnings
from pandas.errors import PerformanceWarning
from Funcs.Utility import _load_data

warnings.simplefilter(action='ignore', category=PerformanceWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

FUNC_PROC = {
    #'Acceleration': _proc_acceleration,
    #'AmbientLight': _proc_ambient_light,
    #'Calorie': _proc_calories,
    #'Distance': _proc_distance,
    #'EDA': _proc_eda,
    'HR': _proc_hr,
    'RRI': _proc_rri,
    #'SkinTemperature': _proc_skin_temperature,
    #'StepCount': _proc_step_count
}


def _process(data_type: str):
    log(f'Begin to processing data: {data_type}')
    
    abbrev = DATA_TYPES[data_type]
    data_raw = _load_data(data_type)
    data_proc = FUNC_PROC[data_type](data_raw)
    result = dict()
    
    if type(data_proc) is dict:
        for k, v in data_proc.items():
            result[f'{abbrev}_{k}'] = v
    else:
        result[abbrev] = data_proc
        
    log(f'Complete processing data: {data_type}')
    return result



#with on_ray(num_cpus=6):
with on_ray():
    jobs = []
    
    func = ray.remote(_process).remote
    
    for data_type in DATA_TYPES:
        job = func(data_type)
        jobs.append(job)

    jobs = ray.get(jobs)
    jobs = reduce(lambda a, b: {**a, **b}, jobs)
    dump(jobs, os.path.join(PATH_INTERMEDIATE, 'proc.pkl'))

    del jobs
    gc.collect()


