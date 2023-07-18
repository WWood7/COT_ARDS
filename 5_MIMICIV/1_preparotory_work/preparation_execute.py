import pandas as pd
from preparation_functions import *
import os

seed = 7
data_path = '/Users/winnwu/projects/emory-hu lab/COT_project/data/mimiciv/'
generate_path = '/Users/winnwu/projects/emory-hu lab/COT_project/generate/mimiciv/'
window_length = 12

#
# first, get the segments of the patients
# get the cohort

case_patients = pd.read_csv(data_path + 'llards_100.csv')
case_patients = case_patients.rename(columns={"charttime": "onset", "stay_id": "icustay_id"})
control_patients = pd.read_csv(data_path + 'llnonards_300_wod.csv')
control_patients = control_patients.rename(columns={"stay_id": "icustay_id"})
patients = {'case': case_patients, 'control': control_patients}

#
# form and save the segments
segs = form_segments(window_length, patients, seed)

folder_path = generate_path + 'segments'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

segs['case'].to_csv(folder_path + '/case_segs.csv')
segs['control'].to_csv(folder_path + '/control_segs.csv')

#
# second