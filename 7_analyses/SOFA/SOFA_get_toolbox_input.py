import pandas as pd
import numpy as np
import datetime

generate_path = '/Users/winnwu/projects/Hu_lab/COT_project/generate/'
data_path = '/Users/winnwu/projects/Hu_lab/COT_project/data/'
SOFA_path = generate_path + 'SOFA/'
segments_path = generate_path + 'segments/'

print('prepare data ...')
test_case_segs = pd.read_csv(segments_path + 'test_case_segs.csv').reset_index(drop=True)[
    ['subject_id', 'hadm_id', 'icustay_id', 'seg_id', 'segstart', 'segend']]
test_control_segs = pd.read_csv(segments_path + 'test_control_segs.csv').reset_index(drop=True)[
    ['subject_id', 'hadm_id', 'icustay_id', 'seg_id', 'segstart', 'segend']]

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

def timedelta_to_hour(time):
    d = 24 * (time.days)
    h = (time.seconds) / 3600
    total_hours = d + h
    return total_hours

sofa_thresh = 9


test_case_array = np.empty((0, 3))
for pat in test_case_segs['seg_id']:
    case_sofa_subset = case_sofa[case_sofa['seg_id'] == pat]
    segend = case_sofa_subset['segend'].iloc[-1]
    case_sofa_subset_sub = case_sofa_subset[
        (case_sofa_subset['endtime'] <= segend)]
    if len(case_sofa_subset_sub) != 0:
            id_list = np.array(case_sofa_subset_sub['icustay_id'])
            timestamp_list = np.array([timedelta_to_hour(segend - val) for val in case_sofa_subset_sub['endtime']])
            hit_list = np.array(case_sofa_subset_sub['sofa_24hours'] >= sofa_thresh).astype(int)
            temparray = np.column_stack((id_list, timestamp_list, hit_list))
            # sort by timestamp
            temparray = temparray[(-temparray[:, 1]).argsort()]

    else:
            temparray = np.array([case_sofa_subset_sub['icustay_id'], 0, 0])
    test_case_array = np.row_stack((test_case_array, temparray))

np.save(SOFA_path + 'SOFA_test_toolbox_input_' + str(sofa_thresh) + '.npy',
        test_case_array, allow_pickle=True)

test_control_array = np.empty((0, 3))
for pat in test_control_segs['seg_id']:
    control_sofa_subset = control_sofa[control_sofa['seg_id'] == pat]
    segend = control_sofa_subset['segend'].iloc[-1]
    control_sofa_subset_sub = control_sofa_subset[
        (control_sofa_subset['endtime'] <= segend)]
    if len(control_sofa_subset_sub) != 0:
            id_list = np.array(control_sofa_subset_sub['icustay_id'])
            timestamp_list = np.array([timedelta_to_hour(segend - val) for val in control_sofa_subset_sub['endtime']])
            hit_list = np.array(control_sofa_subset_sub['sofa_24hours'] >= sofa_thresh).astype(int)
            temparray = np.column_stack((id_list, timestamp_list, hit_list))
            # sort by timestamp
            temparray = temparray[(-temparray[:, 1]).argsort()]
    else:
            temparray = np.array([control_sofa_subset_sub['icustay_id'], 0, 0])
    test_control_array = np.row_stack((test_control_array, temparray))

np.save(SOFA_path + 'SOFA_control_toolbox_input_' + str(sofa_thresh) + '.npy',
        test_control_array, allow_pickle=True)