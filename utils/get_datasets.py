import os
import pandas as pd

# if want get icu folder files
def get_icu_files(filename):
    df = pd.read_csv(os.path.join('icu', filename), compression='gzip',header=0)
    return df

# if want get hosp folder files
def get_hosp_files(filename):
    df = pd.read_csv(os.path.join('hosp', filename), compression='gzip',header=0)
    return df

