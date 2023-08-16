
import pandas as pd
import ast
from generate_online_hitarrays_functions import maplocToTokenID, GenerateOnlineHit, parallel_GenerateOnlineHit


# define the hyperparameters
icu_len = 12
vent_len = 2
prediction_window = 12
delta_value_list = [0.05, 0.1, 0.3, 0.5, 1, 2, 3]
vital_list = ['heartrate', 'sysbp', 'meanbp', 'spo2', 'tempc', 'resprate']
generate_path = '/Users/winnwu/projects/Hu_Lab/COT_project/generate/mimiciv/'
data_path = '/Users/winnwu/projects/Hu_Lab/COT_project/data/mimiciv/'

# read in the segments
case_seg = pd.read_csv(generate_path + 'segments/case_segs.csv').reset_index(drop=True).loc[0:3]
case_seg = case_seg.rename(columns={'segend': 'end'})

control_seg = pd.read_csv(generate_path + 'segments/control_segs.csv').reset_index(drop=True).loc[0:3]
control_seg = control_seg.rename(columns={'segend': 'end'})

print("Read in segments - Complete")


# stack all the ids together
allpathadm_icuid = case_seg.loc[:, ['hadm_id', 'icustay_id', 'INTIME', 'OUTTIME']]

allpathadm_icuid = pd.concat([allpathadm_icuid,
                              control_seg.loc[:, ['hadm_id', 'icustay_id', 'INTIME', 'OUTTIME']]])
print(len(allpathadm_icuid))
print(allpathadm_icuid)
print("Stack in ids - Complete")


# read in vitals for the whole cohort
allvitals = pd.read_csv(data_path + 'allvitals.csv')
allvitals['charttime'] = allvitals['charttime'].astype('datetime64[s]')
allvitals = allvitals.rename(columns={"stay_id": "icustay_id", "heart_rate": "heartrate", "sbp": "sysbp",
                                                 "mbp": "meanbp", "temperature": "tempc", "resp_rate": "resprate"})
print("Read in vitals - Complete")


# read in all lab events with high low marks
alllabs = (pd.read_csv(data_path + 'abnormal_labs_wLH.csv').
           drop(['value', 'icu_intime', 'icu_outtime'], axis=1))
alllabs = alllabs.rename(columns={'charttime': 'CHARTTIME'})
alllabs['CHARTTIME'] = alllabs['CHARTTIME'].astype('datetime64[s]')
alllabs['LABEL'] = alllabs['LABEL'] + '_' + alllabs['LH']
alllabs = pd.merge(alllabs, allpathadm_icuid, how='inner', on='icustay_id').drop_duplicates(keep='first')
alllabs['INTIME'] = alllabs['INTIME'].astype('datetime64[s]')
alllabs['OUTTIME'] = alllabs['OUTTIME'].astype('datetime64[s]')
alllabs = alllabs.loc[(alllabs['INTIME'] <= alllabs['CHARTTIME']) & (alllabs['OUTTIME'] >= alllabs['CHARTTIME'])]
print("Read in lab events - Complete")

# read in all the ventilation events
allvents = pd.read_csv(data_path + 'abnormal_vent_m_s.csv')
allvents['charttime'] = pd.to_datetime(allvents['charttime'])
allvents = allvents.rename(columns={'stay_id': 'icustay_id'})
myaddedlabventduration = pd.read_csv(data_path+'mimiciv_ventilation.csv')
myaddedlabventduration = myaddedlabventduration.reset_index(drop=True)
myaddedlabventduration = myaddedlabventduration.rename(columns={'stay_id': 'icustay_id'})
myaddedlabventduration['starttime'] = myaddedlabventduration['starttime'].astype('datetime64[s]')
print(myaddedlabventduration['starttime'].iloc[0])
myaddedlabventduration['endtime'] = myaddedlabventduration['endtime'].astype('datetime64[s]')
print("Read in ventillation events - Complete")


# read in the demographics
demoevents = pd.read_csv(data_path + 'demographics.csv')
demoevents = demoevents.drop_duplicates(keep='first')
demoevents = demoevents.rename(columns={'stay_id': 'icustay_id'})
height_weight = demoevents.loc[:, ['icustay_id', 'hadm_id', 'height', 'weight', 'BMI']]
print("Read in demographics - Complete")

# read in the token id locations
tokenid_loc = pd.read_csv(generate_path + 'tokens/matrix/tokenid_loc.csv')
print("Read in token id locations - Complete")


FPR_list = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
for max_FPR in FPR_list:
    print('max_FPR: ', max_FPR)
    with open(generate_path + "superalarm/superalarm_" + str(max_FPR) + ".txt", "r") as f:
        content = f.readlines()
    superalarm = [ast.literal_eval(x.strip()) for x in content][0]
    AlarmSetCode = maplocToTokenID(superalarm, tokenid_loc)
    print("maplocToTokenID - Complete")


    print('generating hitarray for case...', len(case_seg['icustay_id'].unique()))
    GenerateOnlineHit(generate_path, case_seg, 'case',
                               allvitals, vital_list, alllabs, allvents, demoevents,
                               prediction_window, AlarmSetCode, delta_value_list, max_FPR, True,
                      myaddedlabventduration)

    print('generating hitarray for control...', len(control_seg['icustay_id'].unique()))
    GenerateOnlineHit(generate_path, control_seg, 'control',
                      allvitals, vital_list, alllabs, allvents, demoevents,
                      prediction_window, AlarmSetCode, delta_value_list, max_FPR, False,
                      myaddedlabventduration)




