import pandas as pd

path = '/Users/winnwu/projects/Hu_Lab/COT_project/generate/mimiciv/segments/'

# Read in the data
case = pd.read_csv(path + 'case_segs.csv')
control = pd.read_csv(path + 'control_segs.csv')
print('icustay_id:', len(case['icustay_id'].unique()), len(control['icustay_id'].unique()))
print('hadm_id:', len(case['hadm_id'].unique()), len(control['hadm_id'].unique()))