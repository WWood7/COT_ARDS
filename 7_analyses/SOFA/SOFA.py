import pandas as pd
import numpy as np
import datetime

generate_path = '/Users/winnwu/projects/Hu_lab/COT_project/generate/'
segments_path = generate_path + 'segments/'
data_path = '/Users/winnwu/projects/Hu_lab/COT_project/data/'

print('prepare data ...')
test_case_segs = pd.read_csv(segments_path + 'test_case_segs.csv').reset_index(drop=True)[
    ['subject_id', 'hadm_id', 'icustay_id', 'seg_id', 'segstart', 'segend']]
test_control_segs = pd.read_csv(segments_path + 'test_control_segs.csv').reset_index(drop=True)[
    ['subject_id', 'hadm_id', 'icustay_id', 'seg_id', 'segstart', 'segend']]

# give hadm_id to pivoted_sofa
# icustay = pd.read_csv('D:/mimic-iii-clinical-database-1.4/ICUSTAYS.csv')[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']]
# icustay = icustay.rename(columns={'ICUSTAY_ID': 'icustay_id', 'HADM_ID': 'hadm_id'})
# pivoted_sofa = pd.read_csv('D:/pivoted_sofa.csv')
# pivoted_sofa_new = pd.merge(pivoted_sofa, icustay, how="left", on=["icustay_id"])
pivoted_sofa_new = pd.read_csv(data_path + 'pivoted_sofa_new.csv')
pivoted_sofa_new = pivoted_sofa_new.rename(columns={'SOFA_24hours': 'sofa_24hours'})
print(len(pivoted_sofa_new))
print(pivoted_sofa_new.head())

case_sofa = pd.merge(pivoted_sofa_new[['hadm_id', 'icustay_id', 'endtime', 'sofa_24hours']],
                     test_case_segs, how="inner", on=["icustay_id"]).reset_index(drop=True)
case_sofa['segstart'] = pd.to_datetime(case_sofa['segstart'])
case_sofa['segend'] = pd.to_datetime(case_sofa['segend'])
case_sofa['endtime'] = pd.to_datetime(case_sofa['endtime'])
control_sofa = pd.merge(pivoted_sofa_new[['hadm_id', 'icustay_id','endtime', 'sofa_24hours']],
                        test_control_segs, how="inner", on=["icustay_id"]).reset_index(drop=True)
control_sofa['segstart'] = pd.to_datetime(control_sofa['segstart'])
control_sofa['segend'] = pd.to_datetime(control_sofa['segend'])
control_sofa['endtime'] = pd.to_datetime(control_sofa['endtime'])


sofa_thresh = 7
TP = 0
for pat in test_case_segs['seg_id']:
    case_sofa_subset = case_sofa[case_sofa['seg_id'] == pat]
    segstart = case_sofa_subset['segstart'].iloc[-1]
    segend = case_sofa_subset['segend'].iloc[-1]
    case_sofa_subset_sub =case_sofa_subset[
        (case_sofa_subset['endtime'] <= segend)
        & (case_sofa_subset['endtime'] >= segstart)]
    if len(case_sofa_subset_sub) != 0:
            case_sofa_subset_thresh = case_sofa_subset_sub[case_sofa_subset_sub['sofa_24hours']>=sofa_thresh]
            if len(case_sofa_subset_thresh) != 0:
                TP = TP+1
print('tpr:', TP/len(test_case_segs))
FP = 0
for pat in test_control_segs['seg_id']:
    control_sofa_subset = control_sofa[control_sofa['seg_id']==pat]
    segstart = control_sofa_subset['segstart'].iloc[-1]
    segend =control_sofa_subset['segend'].iloc[-1]
    # control_sofa_subset_sub =control_sofa_subset[
    #     (control_sofa_subset['endtime']<=(segend+datetime.timedelta(hours=-2))) &
    #     (control_sofa_subset['endtime']>=(segstart+datetime.timedelta(hours=-2)))]
    control_sofa_subset_sub =control_sofa_subset[
        (control_sofa_subset['endtime'] <= segend) &
        (control_sofa_subset['endtime'] >= segstart)]
    if len(control_sofa_subset_sub)!=0:
            control_sofa_subset_thresh = control_sofa_subset_sub[control_sofa_subset_sub['sofa_24hours']>=sofa_thresh]
            if len(control_sofa_subset_thresh) != 0:
                FP = FP+1
print('fpr:', FP / len(test_control_segs))
PPV = TP / (TP + FP)
# F1 scores
F1 = 2 * (PPV * TP/len(test_case_segs)) / (PPV + TP/len(test_case_segs))
print(PPV, F1, sofa_thresh)