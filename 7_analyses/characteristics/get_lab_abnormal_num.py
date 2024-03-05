import pandas as pd
import numpy as np

data_path = '/Users/winnwu/projects/emory-hu lab/COT_project/data/'
generate_path = '/Users/winnwu/projects/emory-hu lab/COT_project/generate/'

# read in the lab events
lab_events_wLH = pd.read_csv(data_path + 'abnormal_labs_wLH.csv')
# read in segments
train_case_segs = pd.read_csv(generate_path + 'train_case_segs.csv')
