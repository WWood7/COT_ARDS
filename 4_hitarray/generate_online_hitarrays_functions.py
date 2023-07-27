from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from trend_processing.GetL1TrendFeatures import plot_l1_trend_fits
from datetime import datetime

def GetVitalToken4OnePiece(generate_path, PatID, subvital_data, vitalname, vital_digit,
                           delta_value_list):
    subvital_tokenlist = []

    subvital_data_feature_table = pd.DataFrame(columns=['icustay_id', 'segment_num',
                                                        'slope_pos_max',
                                                        'slope_neg_max',
                                                        'slope_pos_duration_percent',
                                                        'slope_neg_duration_percent',
                                                        'terminal_max', 'terminal_min',
                                                        'DTposdur1', 'DTnegdur1', 'DTterminal1', 'DTslope1',
                                                        'DTposdur2', 'DTnegdur2', 'DTterminal2', 'DTslope2'])
    for delta_value in delta_value_list:
        segment_num, slope, slope_pos_percent, slope_pos_duration_percent, slope_neg_percent, \
        slope_neg_duration_percent, kink_value, DT_Duration1, DT_terminal1, DT_slope1, left_half_slope, \
        right_half_slope, left_half_dom_dur, \
        right_half_dom_dur, left_half_dom_terminal, right_half_dom_terminal, left_half_dom_slope, \
        right_half_dom_slope, DT_Duration2, DT_terminal2, DT_slope2 \
            = plot_l1_trend_fits(PatID, subvital_data, delta_values=delta_value)
        print("After plot_l1_trend_fits")
        neg_slope = [abs(val) for val in slope if val < 0]
        pos_slope = [val for val in slope if val > 0]

        tempdata = pd.DataFrame({'icustay_id': [PatID], 'segment_num': [segment_num],
                                 'slope_pos_max': [np.max(pos_slope) if len(pos_slope) != 0 else 0],
                                 'slope_neg_max': [np.max(neg_slope) if len(neg_slope) != 0 else 0],
                                 'slope_pos_duration_percent': [slope_pos_duration_percent],
                                 'slope_neg_duration_percent': [slope_neg_duration_percent],
                                 'terminal_max': [np.max(kink_value)], 'terminal_min': [np.min(kink_value)],
                                 'DTposdur1': [DT_Duration1 if DT_slope1 != np.nan and DT_slope1 > 0 else 0],
                                 'DTnegdur1': [DT_Duration1 if DT_slope1 != np.nan and DT_slope1 < 0 else 0],
                                 'DTterminal1': [DT_terminal1],
                                 'DTslope1': [DT_slope1],
                                 'DTposdur2': [DT_Duration2 if DT_slope2 != np.nan and DT_slope2 > 0 else 0],
                                 'DTnegdur2': [DT_Duration2 if DT_slope2 != np.nan and DT_slope2 < 0 else 0],
                                 'DTterminal2': [DT_terminal2],
                                 'DTslope2': [DT_slope2],
                                 })
        subvital_data_feature_table = pd.concat([subvital_data_feature_table, tempdata], ignore_index=True)

    vital_top40 = pd.read_csv(generate_path + 'l1trend_features/top_features/' + vitalname + '_TOP40.csv').reset_index(drop=True)
    # compare the value of the feature with the thresholds stored in the table, if it exceeds, add it into the tokens
    for i in range(len(vital_top40)):
        delta = vital_top40['delta'][i]
        predictor = vital_top40['predictors'][i]
        value = vital_top40['value'][i]
        label = vital_top40['label'][i]
        trend_feature = subvital_data_feature_table.loc[:, ['icustay_id', predictor]].iloc[
            delta_value_list.index(delta)]
        subvital_data_token = []
        # forming a dataframe seems unnecessary?
        if label == 1:
            if trend_feature[predictor] >= value:
                subvital_data_token = pd.DataFrame({'icustay_id': [PatID], predictor: [value]})
        else:
            if trend_feature[predictor] <= value:
                subvital_data_token = pd.DataFrame({'icustay_id': [PatID], predictor: [value]})

        if len(subvital_data_token) != 0:
            subvital_tokenlist.extend([10000 + vital_digit * 1000 + i + 1])
    return subvital_tokenlist


def maplocToTokenID(superalarm, tokenid_loc):
    AlarmSetCode = []
    locID = tokenid_loc.loc[:, ['token_id', 'token_loc']]
    locID['token_loc'] = locID['token_loc'] + 1
    for patterns in superalarm:
        subAlarmSetCode = [locID['token_id'].loc[locID['token_loc'] == val].values[0] for val in patterns]
        AlarmSetCode.append(subAlarmSetCode)
    return AlarmSetCode


def timedelta_to_hour(time):
    d = 24 * (time.days)
    h = (time.seconds) / 3600
    total_hours = d + h
    return total_hours


def GetHitArray(TokenArray, TokenTimeArray, AlarmSetCode):
    TotalNumberofAlarmPatterns = len(AlarmSetCode)
    evaltimes = len(TokenArray)

    HitT_sparse = list(TokenArray.keys())
    HitArray_sparse = np.zeros((TotalNumberofAlarmPatterns, evaltimes))
    HitArrayOR_sparse = np.zeros(evaltimes)
    for i in range(TotalNumberofAlarmPatterns):
        for j in range(len(HitT_sparse)):
            # get the tokens that are triggered at "this" time point
            t = TokenTimeArray[HitT_sparse[j]]
            last_tokens_idx = np.where(t == HitT_sparse[j])
            last_tokens = np.array(TokenArray[HitT_sparse[j]])[last_tokens_idx]
            if set(AlarmSetCode[i]).issubset(set(TokenArray[HitT_sparse[j]])):
                if len(list(set(last_tokens).intersection(set(AlarmSetCode[i])))) != 0:
                    HitArray_sparse[i][j] = 1
                else:
                    HitArray_sparse[i][j] = 0
            else:
                HitArray_sparse[i][j] = 0
            HitArrayOR_sparse[j] = HitArrayOR_sparse[j] or HitArray_sparse[i][j]  # HitArrayOR|HitArray[i]
    sparseHitArray_sparse = csr_matrix(HitArray_sparse)

    return sparseHitArray_sparse, HitArrayOR_sparse, HitT_sparse


def GetData4Patient(DataArray, eventDateTime, iscase):
    # DataArray: columns:'patientid', 'datacharttime', 'dataid', 'datalabel', 'datavalue'
    # eventDateTime = datetime.strptime(eventDateTime, '%Y-%m-%d %H:%M:%S')
    DataArray['TimeToStart'] = DataArray['datacharttime']
    DataArray['datacharttime'] = DataArray['datacharttime'].astype('datetime64[s]')
    patientAlarmArray = []
    RelativeDTofEvent = []
    startalarmcharttime = []
    if iscase:
        # print(eventDateTime)
        # time relative to the CA event in minutes
        # RX: this is for sure the duration from code blue, so larger value means
        # earlier position in timeline for the alarm code
        dt = np.array([timedelta_to_hour(eventDateTime - val) for val in DataArray['datacharttime']])
        # print(dt)
        # RX: this filters out alarm after code blue, since duration will
        # be negative
        iTime = np.where(dt >= 0)
        if len(iTime) != 0:
            aAlarmArray = DataArray.iloc[iTime].reset_index(drop=True)
            t = aAlarmArray['datacharttime']
            # RX: this is important, the duration here is the duration from
            # the first data in timeline, not duration from code blue!!
            tt = np.sort([timedelta_to_hour(val) for val in (t - min(t))])
            # print(tt)
            tidx = np.argsort([timedelta_to_hour(val) for val in (t - min(t))])
            # print(tidx)
            # order the alarms by time
            patientAlarmArray = aAlarmArray.iloc[tidx].reset_index(drop=True)
            # print(patientAlarmArray)
            patientAlarmArray['TimeToStart'] = tt
            RelativeDTofEvent = timedelta_to_hour(eventDateTime - min(t))
            startalarmcharttime = min(t)
    else:
        t = DataArray['datacharttime']
        tt = np.sort([timedelta_to_hour(val) for val in (t - min(t))])
        tidx = np.argsort([timedelta_to_hour(val) for val in (t - min(t))])
        # order the alarms by time
        patientAlarmArray = DataArray.iloc[tidx].reset_index(drop=True)
        patientAlarmArray['TimeToStart'] = tt
        RelativeDTofEvent = np.nan
        startalarmcharttime = min(t)
    # print(patientAlarmArray)
    # print(RelativeDTofEvent)
    return patientAlarmArray, RelativeDTofEvent, startalarmcharttime


def GetDataArray(endtime, allvitals, vital_list, alllabs, allvents, demo):
    DataArray = pd.DataFrame(columns=['patientid', 'datacharttime', 'dataid', 'datalabel', 'datavalue'])
    # endtime = datetime.strptime(endtime, "%Y-%m-%d %H:%M:%S")
    # first get lab events
    suballlabs = alllabs.loc[alllabs['CHARTTIME'] <= endtime]
    if len(suballlabs) != 0:
        patientid = suballlabs['icustay_id'].values
        CHARTTIME = suballlabs['CHARTTIME'].values
        ITEMID = suballlabs['ITEMID'].values
        LABEL = suballlabs['LABEL'].values
        VALUE = suballlabs['VALUE'].values
        tempdata = pd.DataFrame({'patientid': patientid, 'datacharttime': CHARTTIME, 'dataid': ITEMID,
                                 'datalabel': LABEL, 'datavalue': VALUE})
        DataArray = pd.concat([DataArray, tempdata], axis=0, ignore_index=True)

    # second, extract the vital signs
    suballvitals = allvitals.loc[allvitals['charttime'] <= endtime]
    if len(suballvitals) != 0:
        vital_cnt = 0
        for vitalname in vital_list:
            vital_cnt = vital_cnt + 1
            subvital = suballvitals[['icustay_id', 'charttime', vitalname]].dropna(axis=0, how='any', inplace=False)
            if len(subvital) != 0:
                patientid = subvital['icustay_id'].values
                CHARTTIME = subvital['charttime'].values
                ITEMID = [vital_cnt for i in range(len(subvital))]
                LABEL = [vitalname for i in range(len(subvital))]
                VALUE = subvital[vitalname].values
                tempdata = pd.DataFrame({'patientid': patientid, 'datacharttime': CHARTTIME, 'dataid': ITEMID,
                                         'datalabel': LABEL, 'datavalue': VALUE})
                DataArray = pd.concat([DataArray, tempdata], axis=0, ignore_index=True)

    # now get the ventilation data
    subvents = allvents.loc[allvents['charttime'] <= endtime]
    if len(subvents) != 0:
        patientid = subvents['icustay_id'].values
        CHARTTIME = subvents['charttime'].values
        ITEMID = subvents['tokenid'].values
        LABEL = subvents['label'].values
        VALUE = subvents['LH'].values
        tempdata = pd.DataFrame({'patientid': patientid, 'datacharttime': CHARTTIME, 'dataid': ITEMID,
                                 'datalabel': LABEL, 'datavalue': VALUE})
        DataArray = pd.concat([DataArray, tempdata], axis=0, ignore_index=True)

    # now get the demographic data
    demo_list = ['age', 'gender', 'ethnicity', 'height', 'BMI']
    suballdemos = demo
    if len(suballdemos) != 0:
        demo_cnt = 0
        for demoname in demo_list:
            demo_cnt = demo_cnt + 1
            subdemos = suballdemos.loc[:, ['icustay_id', 'end', demoname]].drop_duplicates(keep='first')
            if len(subdemos) != 0:
                patientid = subdemos['icustay_id'].values
                CHARTTIME = endtime
                ITEMID = [300000 + demo_cnt for i in range(len(subdemos))]
                LABEL = [demoname for i in range(len(subdemos))]
                VALUE = subdemos[demoname].values
                tempdata = pd.DataFrame({'patientid': patientid, 'datacharttime': CHARTTIME, 'dataid': ITEMID,
                                         'datalabel': LABEL, 'datavalue': VALUE})
                DataArray = pd.concat([DataArray, tempdata], axis=0, ignore_index=True)

    DataArray.reset_index(drop=True, inplace=True)
    return DataArray


def GetTokenArray4Patient(generate_path, patientDataArray, RelativeDTofEvent, PatID,
                          delta_value_list, prediction_window, vital_list, myaddedlabventduration,
                          maxDurationinMins=np.nan):
    TokenArray = {}
    TokenTimeArray = {}
    t = patientDataArray['TimeToStart'].values
    charttime_start = patientDataArray['datacharttime'].iloc[0]
    myaddedlabventduration['startTostart'] = \
        myaddedlabventduration['starttime'].apply(lambda x: (x - charttime_start).total_seconds()/(60*60))
    myaddedlabventduration['endTostart'] = \
        myaddedlabventduration['endtime'].apply(lambda x: (x - charttime_start).total_seconds()/(60*60))

    # print(t)
    if np.isnan(maxDurationinMins) == False:  # the threshold of duration after the first event that
        # events after this are considered valid
        # control
        if (np.isnan(RelativeDTofEvent)):
            t = t[t <= maxDurationinMins]
        else:
            t = t[(t <= RelativeDTofEvent) & (t >= (RelativeDTofEvent - maxDurationinMins))]

    vital_list_new = [i for i in vital_list if
                      len(patientDataArray['TimeToStart'].loc[patientDataArray['datalabel'] == i]) != 0]
    vital_starttime_list = [
        patientDataArray['TimeToStart'].loc[patientDataArray['datalabel'] == i].reset_index(drop=True)[0] for i in
        vital_list_new]

    evaltime_list = np.sort(list(set(t)))
    print("Eval time list length: %d" % len(evaltime_list))
    # process the static demo tokens first
    demo_list = []
    for demoid in patientDataArray['dataid'][
        (patientDataArray['dataid'] >= 300000) & (patientDataArray['dataid'] < 400000)].unique():
        subonedemo = patientDataArray[patientDataArray['dataid'] == demoid].reset_index(drop=True).iloc[-1]
        if demoid == 300001:
            age = subonedemo['datavalue']
            if age >= 18 and age < 45:
                demo_list.extend([300010])
            elif age >= 45 and age < 65:
                demo_list.extend([300011])
            else:
                demo_list.extend([300012])
        elif demoid == 300002:
            gender = subonedemo['datavalue']
            if gender == 'F':
                demo_list.extend([300020])
            else:
                demo_list.extend([300021])
        elif demoid == 300003:
            ethnicity = subonedemo['datavalue']
            if ethnicity == 'WHITE':
                demo_list.extend([300030])
            elif ethnicity == 'BLACK':
                demo_list.extend([300031])
            elif ethnicity == 'ASIAN':
                demo_list.extend([300032])
            elif ethnicity == 'HISPANIC':
                demo_list.extend([300033])
            else:
                demo_list.extend([300034])
        elif demoid == 300004:
            height = subonedemo['datavalue']
            if height < 150:
                demo_list.extend([300040])
            elif height >= 150 and height < 160:
                demo_list.extend([300041])
            elif height >= 160 and height < 170:
                demo_list.extend([300042])
            elif height >= 170 and height < 180:
                demo_list.extend([300043])
            elif height >= 180 and height < 190:
                demo_list.extend([300044])
            else:
                demo_list.extend([300045])
        else:
            BMI = subonedemo['datavalue']
            if BMI < 18.5:
                demo_list.extend([300050])
            elif BMI >= 18.5 and BMI < 25:
                demo_list.extend([300051])
            elif BMI >= 25 and BMI < 30:
                demo_list.extend([300052])
            elif BMI >= 30 and BMI < 35:
                demo_list.extend([300053])
            else:
                demo_list.extend([300054])

    # process all the other tokens and add demo tokens into the lists
    for evaltime in evaltime_list:
        print("Eval time: %s" % evaltime)
        idx_vital = np.where(((evaltime - t) >= 0) & ((evaltime - t) <= prediction_window))
        subvital_vent_Array = patientDataArray.iloc[idx_vital]
        idx_lab = np.where(((evaltime - t) >= 0) & ((evaltime - t) <= 24))
        sublab_Array = patientDataArray.iloc[idx_lab]

        token_list = []
        tokentime_list = []
        # for lab token, pick the latest one for each lab token
        for labid in sublab_Array['dataid'].loc[sublab_Array['dataid'] > 400000].unique():
            subonelab = sublab_Array.loc[sublab_Array['dataid'] == labid].reset_index(drop=True).iloc[-1]
            token_list.extend([subonelab['dataid']])
            tokentime_list.extend([subonelab['TimeToStart']])
        # for vital tokens
        print("Vital list new length: %d" % (len(vital_list_new)))
        for i in range(len(vital_list_new)):
            vitalname = vital_list_new[i]
            vital_starttime = vital_starttime_list[i]
            # if there exists a complete prediction window for the vital sign, read in the data right from the beginning
            # maybe better to filter out the events that are out of the window here than later?
            if (vital_starttime <= (evaltime - prediction_window)) and ((evaltime - prediction_window) >= 0):
                vital_data = patientDataArray.loc[(patientDataArray['TimeToStart'] >= vital_starttime)
                                                  & (patientDataArray['TimeToStart'] <= evaltime)
                                                  & (patientDataArray['datalabel'] == vitalname)].reset_index(drop=True)
                vital_data['patientid'] = pd.to_numeric(vital_data['patientid'])
                vital_data['dataid'] = pd.to_numeric(vital_data['dataid'])
                vital_data['datavalue'] = pd.to_numeric(vital_data['datavalue'])
                vital_digit = vital_data['dataid'][0]
                # if the current event is not a corresponding vital event, just assume the corresponding vital value is
                # the same as the last one
                if vital_data['TimeToStart'].iloc[-1] != evaltime:
                    evalcharttime = \
                        patientDataArray['datacharttime'].loc[patientDataArray['TimeToStart'] == evaltime].reset_index(
                            drop=True)[0]
                    # vital_data = vital_data.append(
                    #     pd.DataFrame({'patientid': [PatID], 'datacharttime': [evalcharttime], 'TimeToStart': [evaltime],
                    #                   'dataid': [vital_digit], 'datalabel': [vitalname], 'datavalue': [np.nan]}),
                    #     ignore_index=True)
                    tempdata = pd.DataFrame(
                        {'patientid': [PatID], 'datacharttime': [evalcharttime], 'TimeToStart': [evaltime],
                         'dataid': [vital_digit], 'datalabel': [vitalname], 'datavalue': [np.nan]})
                    vital_data = pd.concat([vital_data, tempdata], axis=0, ignore_index=True)
                # print(vital_data['datacharttime'])
                vital_resample = vital_data.resample('60min', on='datacharttime').median().reset_index(drop=False)
                imputed_subvital_resample = vital_resample.ffill(axis=0)
                # print(imputed_subvital_resample['datacharttime'])
                # check whether datacharttime has changed-yes
                imputed_subvital_resample['TimeToEval'] = imputed_subvital_resample['datacharttime'].iloc[-1] - \
                                                          imputed_subvital_resample['datacharttime']
                imputed_subvital_resample['TimeToEval'] = pd.DataFrame(
                    {'TimeToEval': [timedelta_to_hour(val) for val in imputed_subvital_resample['TimeToEval']]})
                # only keep the data that lies in the prediction window
                subvital_data = imputed_subvital_resample['datavalue'][(imputed_subvital_resample['TimeToEval'] >= 0)
                                                                       & (imputed_subvital_resample[
                                                                              'TimeToEval'] < prediction_window)].values
                print("Before GetVitalToken4OnePiece")
                subvital_tokenlist = GetVitalToken4OnePiece(generate_path, PatID, subvital_data, vitalname, vital_digit,
                                                            delta_value_list)
                #subvital_tokenlist = [11004, 11008]
                subvital_tokentimelist = [evaltime for tok in subvital_tokenlist]
                print(subvital_tokenlist)
                if len(subvital_tokenlist) != 0:
                    token_list.extend(subvital_tokenlist)
                    tokentime_list.extend(subvital_tokentimelist)
        #ticTocTimer.tic()
        print("After vital_list_new loop") # slow, takes 1.7s
        # for abnormal ventilation tokens, pick the latest one for each vent tokens
        for ventid in subvital_vent_Array['dataid'].loc[
            (subvital_vent_Array['dataid'] >= 200000) & (subvital_vent_Array['dataid'] < 300000)].unique():
            subonevent = subvital_vent_Array[subvital_vent_Array['dataid'] == ventid].reset_index(drop=True).iloc[-1]
            # if the event is labeled as abnormal
            if subonevent['datavalue'] != 'N':
                token_list.extend([subonevent['dataid']])
                tokentime_list.extend([subonevent['TimeToStart']])
        # for vent delta tokens (transition of ventilation status)
        vent_data = patientDataArray[
            (patientDataArray['dataid'] >= 200000) & (patientDataArray['dataid'] < 300000)].reset_index(drop=True)
        vent_data['originalventid'] = pd.DataFrame(
            {'originalventid': [str(round(val))[:-1] for val in vent_data['dataid']]})
        delta_vent_idx = np.where((evaltime - vent_data['TimeToStart'].values) >= 0)
        delta_vent_data = vent_data.iloc[delta_vent_idx]
        for vent_item_id in delta_vent_data['originalventid'].unique():
            subvent = delta_vent_data.loc[delta_vent_data['dataid'] == vent_item_id].reset_index(drop=True)
            if len(subvent) >= 2:
                subventtemp = subvent[subvent['TimeToStart'] >= evaltime - prediction_window]
                if len(subventtemp) != 0:
                    ventlast1 = subvent.iloc[-1]
                    ventlast2 = subvent.iloc[-2]
                    token_list.extend([round(ventlast2['dataid'] * 1000000 + ventlast1['dataid'])])
                    tokentime_list.extend([ventlast1['TimeToStart']])
        # for vent intubation duration tokens  ## slowest
       
        # only include the patients who are already intubated and the intubation is not over
        subventduration = myaddedlabventduration.loc[(myaddedlabventduration['startTostart'] <= evaltime)
                                                     & (myaddedlabventduration['endTostart'] >= evaltime)]
        if len(subventduration) != 0:
            intub_duration = round((evaltime - subventduration['startTostart'].values[0]) / 24)
            if intub_duration >= 20:
                intub_duration = 20
            token_list.extend([round(21 * 10000 + intub_duration)])
            tokentime_list.extend([evaltime])
        print("Before for demo tokens")
        # for demo tokens
        token_list.extend(demo_list)
        tokentime_list.extend([evaltime for token in demo_list])

        if len(token_list) != 0:
            TokenArray[evaltime] = token_list
            TokenTimeArray[evaltime] = tokentime_list
    # TokenArray:  {evaltime:[tokenid, tokenid,,,,], evaltime:[...],,,,,}
    return TokenArray, TokenTimeArray


def process_GenerateOnlineHit(patlist, patient_set_name, patient_set, allvitals, vital_list, alllabs, allvents,
                              demoevents,
                              iscase, generate_path, delta_value_list, prediction_window,
                              myaddedlabventduration, AlarmSetCode):
    if os.path.exists(generate_path + 'tokenarray/' + patient_set_name + '_TokenArray_dict.npy'):
        case_TokenArray_dict = np.load(generate_path + 'tokenarray/' + patient_set_name + '_TokenArray_dict.npy',
                                       allow_pickle=True).item()
        case_TokenTimeArray_dict = np.load(
            generate_path + 'tokenarray/' + patient_set_name + '_TokenTimeArray_dict.npy', allow_pickle=True).item()
    else:
        case_TokenArray_dict = {}
        case_TokenTimeArray_dict = {}
    case_HitArray_sparse_dict = {}
    case_toolbox_input_sparse = []
    print("TokenArray and TokenTimeArray created")

    # prune the demographic data a little bit
    patientset = patient_set.loc[:, ['hadm_id', 'icustay_id', 'end']]
    demo = pd.merge(demoevents, patientset, how="inner", on=["icustay_id"]).drop_duplicates(keep='first')
    print("Prune patient set and merge with demo events")
    for PatID in patlist:
        print('PatID', PatID)
        # first, extract data from all modules for this patient
        endtime = patientset.loc[patientset['icustay_id'] == PatID]['end'].reset_index(drop=True)[0]
        DataArray = GetDataArray(endtime, allvitals, vital_list, alllabs, allvents, demo)
        print("DataArray created")
        # order the event dataframe based on time and get the relative time to the end of segments
        # and also the relative time to the first data(patientDataArray['datacharttime'])
        patientDataArray, RelativeDTofEvent, startdatacharttime = GetData4Patient(DataArray, PatID, patient_set, iscase)
        print("patientDataArray, RelativeDTofEvent, startdatacharttime created")
        if os.path.exists(generate_path + 'tokenarray/' + patient_set_name + '_TokenArray_dict.npy'):
            TokenArray = case_TokenArray_dict[PatID]
            TokenTimeArray = case_TokenTimeArray_dict[PatID]
        else:
            # Slow
            TokenArray, TokenTimeArray = GetTokenArray4Patient(generate_path, patientDataArray,
                                                               RelativeDTofEvent, PatID,
                                                               delta_value_list, prediction_window, vital_list,
                                                               myaddedlabventduration.loc[myaddedlabventduration['icustay_id'] == PatID])
            case_TokenArray_dict[PatID] = TokenArray
            case_TokenTimeArray_dict[PatID] = TokenTimeArray
        print("TokenArray, TokenTimeArray created")
        # From this TokenArray, we can get two kinds of HitArray, in one we record if the lastly occurred tokens are
        # part of any alarm set, in the other we record if the tokens are part of a specific alarm set
        sparseHitArray_sparse, HitArrayOR_sparse, HitT_sparse = GetHitArray(TokenArray, TokenTimeArray, AlarmSetCode)
        print("HitArray created")
        case_TokenArray_dict[PatID] = TokenArray
        case_TokenTimeArray_dict[PatID] = TokenTimeArray

        case_HitArray_sparse_dict[PatID] = {'sparseHitArray': sparseHitArray_sparse, 'HitT': HitT_sparse}
        Subcase_toolbox_input_sparse = np.zeros((len(HitT_sparse), 3))
        Subcase_toolbox_input_sparse[:, 0] = PatID
        Subcase_toolbox_input_sparse[:, 2] = HitArrayOR_sparse
        if iscase:
            Subcase_toolbox_input_sparse[:, 1] = [RelativeDTofEvent - val for val in HitT_sparse]
        else:
            Subcase_toolbox_input_sparse[:, 1] = [timedelta_to_hour(
                patient_set['end'][patient_set['icustay_id'] == PatID].reset_index(drop=True)[
                    0] - startdatacharttime) - val
                                                  for val in HitT_sparse]

        if len(case_toolbox_input_sparse) == 0:
            case_toolbox_input_sparse = Subcase_toolbox_input_sparse
        else:
            case_toolbox_input_sparse = np.vstack((case_toolbox_input_sparse, Subcase_toolbox_input_sparse))
    return case_HitArray_sparse_dict, case_TokenArray_dict, case_TokenTimeArray_dict, case_toolbox_input_sparse


def parallel_GenerateOnlineHit(generate_path, patient_set, patient_set_name,
                               allvitals, vital_list, alllabs, allvents, demoevents, height_weight,
                               prediction_window, AlarmSetCode, delta_value_list, max_FPR, iscase,
                               myaddedlabventduration):

    # Generate hit array for each patient
    case_TokenArray_dict = {}
    case_TokenTimeArray_dict = {}
    case_HitArray_sparse_dict = {}
    case_toolbox_input_sparse = []  # (patient IDs, relative time in hours to event onset, predictions)

    partial_work = partial(process_GenerateOnlineHit, patient_set_name=patient_set_name, patient_set=patient_set,
                           allvitals=allvitals, vital_list=vital_list,
                           alllabs=alllabs, allvents=allvents, demoevents=demoevents,
                           iscase=iscase, generate_path=generate_path, delta_value_list=delta_value_list,
                           prediction_window=prediction_window,
                           myaddedlabventduration=myaddedlabventduration, AlarmSetCode=AlarmSetCode)
    pool = ThreadPool()
    interval = round(len(patient_set['icustay_id'].unique()) / 8)
    if interval != 0:
        patternset_sublist = [patient_set['icustay_id'].unique()[i * interval:(i + 1) * interval] for i in range(7)]
        patternset_sublist.append(patient_set['icustay_id'].unique()[7 * interval:])
    else:
        patternset_sublist = [patient_set['icustay_id'].unique()]
    return patternset_sublist
    results = pool.map(partial_work, patternset_sublist)
    pool.close()
    pool.join()
    for pi in range(len(patternset_sublist)):
        if os.path.exists(generate_path + 'tokenarray/' + patient_set_name + '_TokenArray_dict.npy'):
            print('token array exist!')
        else:
            case_TokenArray_dict.update(results[pi][1])
            case_TokenTimeArray_dict.update(results[pi][2])
        case_HitArray_sparse_dict.update(results[pi][0])
        if len(case_toolbox_input_sparse) == 0:
            case_toolbox_input_sparse = results[pi][3]
        else:
            case_toolbox_input_sparse = np.vstack((case_toolbox_input_sparse, results[pi][3]))
        print(len(case_toolbox_input_sparse))

    print("Saving...")
    np.save(generate_path + 'tokenarray/' + patient_set_name + '_HitArray_dict_' + str(max_FPR) + '_sparse.npy',
            case_HitArray_sparse_dict, allow_pickle=True)
    np.save(generate_path + 'tokenarray/' + patient_set_name + '_toolbox_input_' + str(max_FPR) + '_sparse.npy',
            case_toolbox_input_sparse, allow_pickle=True)
    if os.path.exists(generate_path + 'tokenarray/' + patient_set_name + '_TokenArray_dict.npy'):
        print('token array exist!')
    else:
        np.save(generate_path + 'tokenarray/' + patient_set_name + '_TokenArray_dict.npy', case_TokenArray_dict,
                allow_pickle=True)
        np.save(generate_path + 'tokenarray/' + patient_set_name + '_TokenTimeArray_dict.npy',
                case_TokenTimeArray_dict,
                allow_pickle=True)


def GenerateOnlineHit(generate_path, patient_set, patient_set_name,
                      allvitals, vital_list, alllabs, allvents, demoevents,
                      prediction_window, AlarmSetCode, delta_value_list, max_FPR, iscase, myaddedlabventduration):
    patient_set['end'] = patient_set['end'].astype('datetime64[ns]')
    # generate hitarray for each patient

    if os.path.exists(generate_path + 'tokenarray/' + patient_set_name + '_TokenArray_dict.npy'):
        case_TokenArray_dict = np.load(generate_path + 'tokenarray/' + patient_set_name + '_TokenArray_dict.npy',
                                       allow_pickle=True).item()
        case_TokenTimeArray_dict = np.load(generate_path + 'tokenarray/' + patient_set_name +
                                           '_TokenTimeArray_dict.npy', allow_pickle=True).item()
    else:
        case_TokenArray_dict = {}
        case_TokenTimeArray_dict = {}
    case_HitArray_sparse_dict = {}
    case_toolbox_input_sparse = []  # (patient IDs, relative time in hours to event onset, predictions)
    case_cnt = 0

    # prune the demographic data a little bit
    patientset = patient_set.loc[:, ['hadm_id', 'icustay_id', 'end']]
    demo = pd.merge(demoevents, patientset, how="inner", on=["icustay_id"]).drop_duplicates(keep='first')

    for PatID in patient_set['icustay_id'].unique():
        print('PatID', PatID)
        # first, extract data from all modules for this patient
        endtime = patientset.loc[patientset['icustay_id'] == PatID]['end'].reset_index(drop=True)[0]
        DataArray = GetDataArray(endtime, allvitals.loc[allvitals['icustay_id'] == PatID],
                                 vital_list,
                                 alllabs.loc[alllabs['icustay_id'] == PatID],
                                 allvents.loc[allvents['icustay_id'] == PatID],
                                 demo.loc[demo['icustay_id'] == PatID])

        # order the event dataframe based on time and get the relative time to the end of segments
        # and also the relative time to the first data(patientDataArray['datacharttime'])
        patientDataArray, RelativeDTofEvent, startdatacharttime = GetData4Patient(DataArray, endtime, iscase)
        if os.path.exists(generate_path + 'tokenarray/' + patient_set_name + '_TokenArray_dict.npy'):
            TokenArray = case_TokenArray_dict[PatID]
            TokenTimeArray = case_TokenTimeArray_dict[PatID]
        else:
            # get all the tokens and evaluate time (for each new data point, evaluate)
            TokenArray, TokenTimeArray = GetTokenArray4Patient(generate_path, patientDataArray, RelativeDTofEvent,
                                                               PatID, delta_value_list, prediction_window, vital_list,
                                                               myaddedlabventduration.
                                                               loc[myaddedlabventduration['icustay_id'] == PatID])
            case_TokenArray_dict[PatID] = TokenArray
            case_TokenTimeArray_dict[PatID] = TokenTimeArray

        # From this TokenArray, we can get two kinds of HitArray, in one we record if the lastly occurred tokens are
        # part of any alarm set, in the other we record if the tokens are part of a specific alarm set
        sparseHitArray_sparse, HitArrayOR_sparse, HitT_sparse = GetHitArray(TokenArray, TokenTimeArray, AlarmSetCode)

        case_HitArray_sparse_dict[PatID] = {'sparseHitArray': sparseHitArray_sparse, 'HitT': HitT_sparse}
        Subcase_toolbox_input_sparse = np.zeros((len(HitT_sparse), 3))
        Subcase_toolbox_input_sparse[:, 0] = PatID
        Subcase_toolbox_input_sparse[:, 2] = HitArrayOR_sparse
        if iscase:
            Subcase_toolbox_input_sparse[:, 1] = [RelativeDTofEvent - val for val in HitT_sparse]
        else:
            Subcase_toolbox_input_sparse[:, 1] = [timedelta_to_hour(
                patient_set['end'][patient_set['icustay_id'] == PatID].reset_index(drop=True)[
                    0] - startdatacharttime) - val
                                                  for val in HitT_sparse]

        if len(case_toolbox_input_sparse) == 0:
            case_toolbox_input_sparse = Subcase_toolbox_input_sparse
        else:
            case_toolbox_input_sparse = np.vstack((case_toolbox_input_sparse, Subcase_toolbox_input_sparse))
        case_cnt = case_cnt + 1
    print("Saving...")
    np.save(generate_path + 'tokenarray/' + patient_set_name + '_HitArray_dict_' + str(max_FPR) + '_sparse.npy',
            case_HitArray_sparse_dict, allow_pickle=True)
    np.save(generate_path + 'tokenarray/' + patient_set_name + '_toolbox_input_' + str(max_FPR) + '_sparse.npy',
            case_toolbox_input_sparse, allow_pickle=True)
    if os.path.exists(generate_path + 'tokenarray/' + patient_set_name + '_TokenArray_dict.npy'):
        print('token array exist!')
    else:
        np.save(generate_path + 'tokenarray/' + patient_set_name + '_TokenArray_dict.npy', case_TokenArray_dict,
                allow_pickle=True)
        np.save(generate_path + 'tokenarray/' + patient_set_name + '_TokenTimeArray_dict.npy', case_TokenTimeArray_dict,
                allow_pickle=True)
