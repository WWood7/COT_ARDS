import pandas as pd
import numpy as np
import datetime
from trend_processing.l1_trend import case_get_trend_feature, control_get_trend_feature
import os


def timedelta_to_hour(time):
    d = 24 * (time.days)
    h = (time.seconds) / 3600
    total_hours = d + h
    return total_hours


def impute_vitals(vitals):
    #
    # resample the vital signals at a 1-hour frequency
    # use forward filling to impute
    # or just use the certain values

    vital_list = ['heartrate', 'sysbp', 'meanbp', 'spo2', 'tempc', 'resprate']
    vital_value_list = [92, 119, 77, 97, 36, 20]
    allvitals = vitals.copy()
    allvitals['charttime'] = pd.to_datetime(allvitals['charttime'])
    icu_unique = allvitals['icustay_id'].unique()
    icu_unique.sort()
    datahadmID = []
    for id in range(len(icu_unique)):
        datahadmID.append(allvitals.loc[allvitals['icustay_id'] == icu_unique[id]])
    for i in range(len(datahadmID)):
        datahadmID[i] = datahadmID[i].sort_values(by=['charttime'])
    print(len(datahadmID), 'encounters')

    global_median = allvitals[vital_list].median()
    for vital_idx in range(len(vital_list)):
        global_median[vital_list[vital_idx]] = vital_value_list[vital_idx]

    final_patients_per_hour = []
    for id in datahadmID:
        id['charttime'] = pd.to_datetime(id['charttime'])
        temp = id.resample('60min', on='charttime').median()
        temp = temp.fillna(axis=0, method='ffill')
        temp = temp.fillna(global_median)
        final_patients_per_hour.append(temp)

    imputed_vitals = pd.concat([obj for obj in final_patients_per_hour])
    imputed_vitals['charttime'] = imputed_vitals.index
    imputed_vitals.reset_index(drop=True)
    return imputed_vitals


def CalculateFeatures(test_case_segs, test_control_segs,
                      imputed_vitals, direction):
    lambda_list = [0.05, 0.1, 0.3, 0.5, 1, 2, 3]
    duration = 12
    method = 'l1'
    vital_list = ['heartrate', 'sysbp', 'meanbp', 'spo2', 'tempc', 'resprate']

    #
    # calculate features
    folder_path = direction + '/l1trend_features'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print("Calculating whole set features...")
    for i in range(len(vital_list)):
        vital_name = vital_list[i]
        vital = imputed_vitals.loc[:, ['hadm_id', 'icustay_id', 'charttime', vital_name]]
        imputed_vital_test_case = pd.merge(vital, test_case_segs.loc[:, ['icustay_id']],
                                           how="inner",
                                           on=["icustay_id"]).drop_duplicates(keep='first')
        imputed_vital_test_control = pd.merge(vital, test_control_segs.loc[:, ['icustay_id']],
                                              how="inner",
                                              on=["icustay_id"]).drop_duplicates(keep='first')

        # get features table
        print("Getting features table...")

        for j in range(len(lambda_list)):
            delta_value = lambda_list[j]

            test_case_feature_table = case_get_trend_feature(imputed_vital_test_case,
                                                             test_case_segs,
                                                             vital_name,
                                                             duration, delta_value, method)
            test_case_feature_table.to_csv(
                folder_path + '/l1trendfeature_' + str(
                    delta_value) + '_' + vital_name + '_test_case.csv')

            test_control_feature_table = control_get_trend_feature(imputed_vital_test_control,
                                                                   test_control_segs,
                                                                   vital_name, duration, delta_value)
            test_control_feature_table.to_csv(
                folder_path + '/l1trendfeature_' + str(
                    delta_value) + '_' + vital_name + '_test_control.csv')


def Getabnomral_labs_wLH(data_path):
    # select abnormal events first
    labevents = pd.read_csv(data_path + 'alllabs.csv')
    labevents = labevents.rename(columns={"stay_id": "icustay_id", "itemid": "ITEMID", "valuenum": "VALUE",
                                          "flag": "FLAG"})
    labevents["VALUE"] = pd.to_numeric(labevents["VALUE"], errors='coerce', downcast='float')
    d_labitems = pd.read_csv(data_path + 'd_labitems.csv')
    d_labitems = d_labitems.rename(columns={"itemid": "ITEMID", "label": "LABEL"})
    print(len(labevents))
    ab_labevents_id = labevents[['ITEMID']][labevents['FLAG'] == 'abnormal'].drop_duplicates(keep='first')
    print(len(ab_labevents_id))

    norm_labevents = labevents.loc[labevents['FLAG'] != 'abnormal']
    # here use the lab events with normal flags to calculate a mean
    ab_labevents_id.loc[:, 'Mean'] = np.nan
    for ITEMID in ab_labevents_id['ITEMID'].unique():
        onelab_mean = norm_labevents['VALUE'].loc[norm_labevents['ITEMID'] == ITEMID].mean()
        ab_labevents_id['Mean'][ab_labevents_id['ITEMID'] == ITEMID] = onelab_mean

    # merge the ids and mean values for the items with the same concepts
    merge_labs = {'bicarbonate': [50882, 50803],
                  'bilirubin': [50884, 50883, 50885],
                  'chloride': [50806, 50902],
                  'eosinophils': [51444, 51347, 51419, 51114],
                  'glucose': [50809, 50931],
                  'hematocrit': [50810, 51221],
                  'hemoglobin': [51222, 50811],
                  'lymphocytes body fluid': [51375, 51427],
                  'ph': [50831, 50820],
                  'ph urine': [51491, 51094],
                  'potassium': [50971, 50822],
                  'sodium': [50983, 50824],
                  'white blood cell count': [51301, 51300],
                  }
    for key in merge_labs.keys():
        merge_mean = 0  # 将所有要merge的mean加起来
        merge_mean_count = 0  # 计数有几个
        for i in range(len(merge_labs[key]) - 1):
            if len(ab_labevents_id.loc[ab_labevents_id['ITEMID'] == merge_labs[key][i + 1]]) != 0:
                temp_mean = ab_labevents_id['Mean'].loc[ab_labevents_id['ITEMID'] == merge_labs[key][i + 1]].unique()[0]
                print(temp_mean)
                if temp_mean != np.nan:
                    print("In")
                    merge_mean_count = merge_mean_count + 1
                    merge_mean = temp_mean + merge_mean
                ab_labevents_id['ITEMID'].loc[ab_labevents_id['ITEMID'] == merge_labs[key][i + 1]] = merge_labs[key][0]
        if merge_mean_count != 0:
            ab_labevents_id['Mean'].loc[ab_labevents_id['ITEMID'] == merge_labs[key][0]] = merge_mean / merge_mean_count
    print(len(ab_labevents_id['ITEMID'].unique()))

    # extract the abnormal lab events with labels
    ab_labevents = labevents.loc[labevents['FLAG'] == 'abnormal']
    ab_labevents_wLabel = pd.merge(
        ab_labevents, d_labitems[['ITEMID', 'LABEL']],
        how="left", on=["ITEMID"]).drop_duplicates(keep='first')
    for key in merge_labs.keys():
        for i in range(len(merge_labs[key]) - 1):
            ab_labevents_wLabel['ITEMID'].loc[ab_labevents_wLabel['ITEMID'] == merge_labs[key][i + 1]] = \
                merge_labs[key][0]
            ab_labevents_wLabel['LABEL'].loc[ab_labevents_wLabel['ITEMID'] == merge_labs[key][0]] = key

    ab_labevents_wLabelMean = pd.merge(
        ab_labevents_wLabel, ab_labevents_id,
        how="left", on=["ITEMID"]).drop_duplicates(keep='first')

    #
    ab_labevents_wLabelMean['originalitemid'] = ab_labevents_wLabelMean['ITEMID'].copy()
    ab_labevents_wLabelMean['ITEMID'] = ab_labevents_wLabelMean['ITEMID'] * 10
    ab_labevents_wLabelMean['LH'] = 'N'
    ab_labevents_wLabelMean["VALUE"] = pd.to_numeric(ab_labevents_wLabelMean["VALUE"], errors='coerce')
    print(len(ab_labevents_wLabelMean))
    print(ab_labevents_wLabelMean.head())
    print(len(ab_labevents_wLabelMean['ITEMID'].unique()))

    count = 0
    r_count = 0
    for ITEMID in ab_labevents_wLabelMean['ITEMID'].unique():
        count = count + 1
        print("COUNT!!!!!!!!!!!!!!!!!!!!!!!!!:", count)
        if np.isnan(ab_labevents_wLabelMean['Mean'].loc[ab_labevents_wLabelMean['ITEMID'] == ITEMID].unique()[0]):
            print(ITEMID, ab_labevents_wLabelMean['LABEL'].loc[ab_labevents_wLabelMean['ITEMID'] == ITEMID].unique()[0])
        else:
            r_count = r_count + 1
            range_mean = ab_labevents_wLabelMean['Mean'].loc[ab_labevents_wLabelMean['ITEMID'] == ITEMID].unique()[0]
            print(range_mean)
            ab_high = ab_labevents_wLabelMean.loc[
                (ab_labevents_wLabelMean['ITEMID'] == ITEMID) & (ab_labevents_wLabelMean['VALUE'] >= range_mean) & (
                        ab_labevents_wLabelMean['FLAG'] == 'abnormal')]
            if len(ab_high) != 0:
                ab_labevents_wLabelMean['LH'].loc[
                    (ab_labevents_wLabelMean['ITEMID'] == ITEMID) & (ab_labevents_wLabelMean['VALUE'] >= range_mean) & (
                            ab_labevents_wLabelMean['FLAG'] == 'abnormal')] = 'High'
                temp1 = ab_labevents_wLabelMean['ITEMID'].loc[
                    (ab_labevents_wLabelMean['ITEMID'] == ITEMID) & (ab_labevents_wLabelMean['VALUE'] >= range_mean) & (
                            ab_labevents_wLabelMean['FLAG'] == 'abnormal')].unique()[0]
                temp2 = ab_labevents_wLabelMean['LABEL'].loc[
                    (ab_labevents_wLabelMean['ITEMID'] == ITEMID) & (ab_labevents_wLabelMean['VALUE'] >= range_mean) & (
                            ab_labevents_wLabelMean['FLAG'] == 'abnormal')].unique()[0]
                # print(temp1, temp2)
                ab_labevents_wLabelMean['ITEMID'].loc[
                    (ab_labevents_wLabelMean['ITEMID'] == ITEMID) & (ab_labevents_wLabelMean['VALUE'] >= range_mean) & (
                            ab_labevents_wLabelMean['FLAG'] == 'abnormal')] = temp1 + 2
                ab_labevents_wLabelMean['LABEL'].loc[
                    (ab_labevents_wLabelMean['ITEMID'] == ITEMID) & (ab_labevents_wLabelMean['VALUE'] >= range_mean) & (
                            ab_labevents_wLabelMean['FLAG'] == 'abnormal')] = temp2 + "_High"
            ab_low = ab_labevents_wLabelMean.loc[(ab_labevents_wLabelMean['ITEMID'] == ITEMID) & (
                    ab_labevents_wLabelMean['VALUE'] < range_mean) & (ab_labevents_wLabelMean['FLAG'] == 'abnormal')]
            if len(ab_low) != 0:
                ab_labevents_wLabelMean['LH'].loc[
                    (ab_labevents_wLabelMean['ITEMID'] == ITEMID) & (ab_labevents_wLabelMean['VALUE'] < range_mean) & (
                            ab_labevents_wLabelMean['FLAG'] == 'abnormal')] = 'Low'
                temp3 = ab_labevents_wLabelMean['ITEMID'].loc[
                    (ab_labevents_wLabelMean['ITEMID'] == ITEMID) & (ab_labevents_wLabelMean['VALUE'] < range_mean) & (
                            ab_labevents_wLabelMean['FLAG'] == 'abnormal')].unique()[0]
                temp4 = ab_labevents_wLabelMean['LABEL'].loc[
                    (ab_labevents_wLabelMean['ITEMID'] == ITEMID) & (ab_labevents_wLabelMean['VALUE'] < range_mean) & (
                            ab_labevents_wLabelMean['FLAG'] == 'abnormal')].unique()[0]
                ab_labevents_wLabelMean['ITEMID'].loc[
                    (ab_labevents_wLabelMean['ITEMID'] == ITEMID) & (ab_labevents_wLabelMean['VALUE'] < range_mean) & (
                            ab_labevents_wLabelMean['FLAG'] == 'abnormal')] = temp3 + 1
                ab_labevents_wLabelMean['LABEL'].loc[
                    (ab_labevents_wLabelMean['ITEMID'] == ITEMID) & (ab_labevents_wLabelMean['VALUE'] < range_mean) & (
                            ab_labevents_wLabelMean['FLAG'] == 'abnormal')] = temp4 + "_Low"

    # print(r_count,"have h or low")
    print(len(ab_labevents_wLabelMean['ITEMID'].unique()))
    ab_labevents_wLabelMean.to_csv(data_path + 'abnormal_labs_wLH.csv')
    return ab_labevents_wLabelMean


def GetLabTokens(test_case_segs, test_control_segs, abnormal_labs_wLH):
    labevents_wLabel = abnormal_labs_wLH
    labevents_wLabel = labevents_wLabel.rename(columns={'charttime': 'CHARTTIME', 'valueuom': 'VALUEUOM'})

    labevents_wLabel['LABELLH'] = labevents_wLabel['LABEL'] + '_' + labevents_wLabel['LH']
    print(len(labevents_wLabel))
    print(len(labevents_wLabel['ITEMID'].unique()))

    test_case = test_case_segs[['hadm_id', 'icustay_id', 'segend', 'segstart', 'starttime']]
    test_control = test_control_segs[
        ['hadm_id', 'icustay_id', 'segend', 'seg_id', 'segstart', 'starttime']]

    test_case_lab = pd.merge(labevents_wLabel, test_case, how="inner", on=["hadm_id", "icustay_id"])\
        .drop_duplicates(keep='first')
    test_case_lab['CHARTTIME'] = pd.to_datetime(test_case_lab['CHARTTIME'])
    test_case_lab['segend'] = pd.to_datetime(test_case_lab['segend'])
    test_case_lab['segstart'] = pd.to_datetime(test_case_lab['segstart'])
    test_case_lab['starttime'] = pd.to_datetime(test_case_lab['starttime'])
    test_control_lab = pd.merge(labevents_wLabel, test_control, how="inner", on=["hadm_id"]).drop_duplicates(
        keep='first')
    test_control_lab['CHARTTIME'] = pd.to_datetime(test_control_lab['CHARTTIME'])
    test_control_lab['segend'] = pd.to_datetime(test_control_lab['segend'])
    test_control_lab['segstart'] = pd.to_datetime(test_control_lab['segstart'])
    test_control_lab['starttime'] = pd.to_datetime(test_control_lab['starttime'])

    # form case lab tokens
    # test set
    print("dealing with test case lab")
    uniperIDs = test_case_lab['icustay_id'].unique()
    uniperIDs.sort()
    datahadmID = []
    test_case_lab_token_list = []
    for id in range(len(uniperIDs)):
        datahadmID.append(test_case_lab[test_case_lab['icustay_id'] == uniperIDs[id]])
    for i in range(len(datahadmID)):
        datahadmID[i] = datahadmID[i][(datahadmID[i]['CHARTTIME'] <= datahadmID[i]['segend'].values[0]) & (
                datahadmID[i]['CHARTTIME'] + datetime.timedelta(hours=24) >= datahadmID[i]['segend'].values[0])]
        if len(datahadmID[i]) > 0:
            datahadmID[i] = datahadmID[i].sort_values(by=['CHARTTIME'])
            for j in datahadmID[i]['ITEMID'].unique():
                temp_onelab = datahadmID[i][datahadmID[i]['ITEMID'] == j]
                test_case_lab_token_series = \
                    temp_onelab[['icustay_id', 'CHARTTIME', 'VALUE', 'VALUEUOM', 'ITEMID', 'LABELLH', 'segend']].iloc[
                        -1]
                test_case_lab_token = pd.DataFrame({'icustay_id': [test_case_lab_token_series['icustay_id']],
                                                    'CHARTTIME': [test_case_lab_token_series['CHARTTIME']],
                                                    'VALUE': [test_case_lab_token_series['VALUE']],
                                                    'VALUEUOM': [test_case_lab_token_series['VALUEUOM']],
                                                    'ITEMID': [test_case_lab_token_series['ITEMID']],
                                                    'LABEL': [test_case_lab_token_series['LABELLH']],
                                                    'segend': [test_case_lab_token_series['segend']]})
                test_case_lab_token['time'] = timedelta_to_hour(
                    pd.to_datetime(test_case_lab_token['segend'].values[0])
                    - pd.to_datetime(test_case_lab_token['CHARTTIME'].values[0]))
                test_case_lab_token_list.append(test_case_lab_token)
    test_case_lab_token_input = pd.concat([obj for obj in test_case_lab_token_list])
    print(len(test_case_lab_token_input))
    test_case_lab_token_input = test_case_lab_token_input.rename(
        columns={"icustay_id": "PatientID", "VALUE": "value", "ITEMID": "token_id", "LABEL": "token"})
    print(len(test_case_lab_token_input['token_id'].unique()))

    # form control lab tokens
    # test set
    print("dealing with test control lab")
    uniperIDs = test_control_lab['seg_id'].unique()
    uniperIDs.sort()
    datahadmID = []
    test_control_lab_token_list = []
    for id in range(len(uniperIDs)):
        datahadmID.append(test_control_lab[test_control_lab['seg_id'] == uniperIDs[id]])
    for i in range(len(datahadmID)):
        datahadmID[i] = datahadmID[i][(datahadmID[i]['CHARTTIME'] <= datahadmID[i]['segend'].values[0]) & (
                datahadmID[i]['CHARTTIME'] + datetime.timedelta(hours=24) >= datahadmID[i]['segend'].values[0])]
        if len(datahadmID[i]) > 0:
            datahadmID[i] = datahadmID[i].sort_values(by=['CHARTTIME'])
            for j in datahadmID[i]['ITEMID'].unique():
                temp_onelab = datahadmID[i][datahadmID[i]['ITEMID'] == j]
                test_control_lab_token_series = \
                    temp_onelab[['seg_id', 'CHARTTIME', 'VALUE', 'VALUEUOM', 'ITEMID', 'LABELLH', 'segend']].iloc[-1]
                test_control_lab_token = pd.DataFrame({'seg_id': [test_control_lab_token_series['seg_id']],
                                                       'CHARTTIME': [test_control_lab_token_series['CHARTTIME']],
                                                       'VALUE': [test_control_lab_token_series['VALUE']],
                                                       'VALUEUOM': [test_control_lab_token_series['VALUEUOM']],
                                                       'ITEMID': [test_control_lab_token_series['ITEMID']],
                                                       'LABEL': [test_control_lab_token_series['LABELLH']],
                                                       'segend': [test_control_lab_token_series['segend']]})
                test_control_lab_token['time'] = timedelta_to_hour(
                    pd.to_datetime(test_control_lab_token['segend'].values[0])
                    - pd.to_datetime(test_control_lab_token['CHARTTIME'].values[0]))
                test_control_lab_token_list.append(test_control_lab_token)
    test_control_lab_token_input = pd.concat([obj for obj in test_control_lab_token_list])
    print(len(test_control_lab_token_input))
    test_control_lab_token_input = test_control_lab_token_input.rename(
        columns={"seg_id": "PatientID", "VALUE": "value", "ITEMID": "token_id", "LABEL": "token"})
    print(len(test_control_lab_token_input['token_id'].unique()))

    return test_case_lab_token_input, \
           test_control_lab_token_input


def GetVentTokens(test_case_segs, test_control_segs,
                  abnormal_vent, vent_token_method):
    abnormal_vent = abnormal_vent.rename(columns={'itemid': 'ITEMID', 'stay_id': 'icustay_id'})
    abnormal_vent['originalitemid'] = abnormal_vent['ITEMID'].copy()
    if vent_token_method == 'all':
        abnormal_vent = abnormal_vent[abnormal_vent['abnormal_flag'] != 0]
    else:
        abnormal_vent = abnormal_vent

    abnormal_vent['labelLH'] = abnormal_vent['label'] + '_' + abnormal_vent['LH']
    abnormal_vent = abnormal_vent.rename(columns={"charttime": "CHARTTIME", 'label': 'LABEL', 'valueuom': 'VALUEUOM'})
    print(len(abnormal_vent))
    print(len(abnormal_vent['tokenid'].unique()))

    test_case = test_case_segs[['hadm_id', 'icustay_id', 'segend', 'segstart', 'starttime']]
    test_control = test_control_segs[
        ['hadm_id', 'icustay_id', 'segend', 'seg_id', 'segstart', 'starttime']]

    test_case_vent = pd.merge(abnormal_vent, test_case, how="inner", on=["icustay_id"]).drop_duplicates(keep='first')
    test_case_vent['CHARTTIME'] = pd.to_datetime(test_case_vent['CHARTTIME'])
    test_case_vent['segend'] = pd.to_datetime(test_case_vent['segend'])
    test_case_vent['segstart'] = pd.to_datetime(test_case_vent['segstart'])
    test_case_vent['starttime'] = pd.to_datetime(test_case_vent['starttime'])
    test_control_vent = pd.merge(abnormal_vent, test_control, how="inner", on=["icustay_id"]).drop_duplicates(
        keep='first')
    test_control_vent['CHARTTIME'] = pd.to_datetime(test_control_vent['CHARTTIME'])
    test_control_vent['segend'] = pd.to_datetime(test_control_vent['segend'])
    test_control_vent['segstart'] = pd.to_datetime(test_control_vent['segstart'])
    test_control_vent['starttime'] = pd.to_datetime(test_control_vent['starttime'])

    if vent_token_method == 'lasttwo':
        # form case vent tokens
        # test set
        print("dealing with test case vent")
        uniperIDs = test_case_vent['icustay_id'].unique()
        uniperIDs.sort()
        datahadmID = []
        test_case_vent_token_list = []
        for id in range(len(uniperIDs)):
            datahadmID.append(test_case_vent[test_case_vent['icustay_id'] == uniperIDs[id]].reset_index(drop=True))
        for i in range(len(datahadmID)):
            datahadmID[i] = datahadmID[i][(datahadmID[i]['CHARTTIME'] <= datahadmID[i]['segend'].values[0]) & (
                    datahadmID[i]['CHARTTIME'] >= datahadmID[i]['starttime'].values[0])]
            if len(datahadmID[i]) > 0:
                datahadmID[i] = datahadmID[i].sort_values(by=['CHARTTIME'])
                for j in datahadmID[i]['ITEMID'].unique():
                    temp_onevent = datahadmID[i][datahadmID[i]['ITEMID'] == j].reset_index(drop=True)
                    if len(temp_onevent) >= 2:
                        test_case_vent_token_series1 = temp_onevent[
                            ['icustay_id', 'ITEMID', 'CHARTTIME', 'tokenid', 'labelLH',
                             'segend']].iloc[-1]
                        test_case_vent_token_series2 = \
                            temp_onevent[
                                ['icustay_id', 'ITEMID', 'CHARTTIME', 'tokenid', 'labelLH',
                                 'segend']].iloc[-2]
                        test_case_vent_token = pd.DataFrame({'icustay_id': [test_case_vent_token_series1['icustay_id']],
                                                             'ITEMID': [
                                                                 test_case_vent_token_series1['ITEMID']],
                                                             'CHARTTIME': [test_case_vent_token_series1['CHARTTIME']],
                                                             'tokenid': [
                                                                 round(
                                                                     test_case_vent_token_series2['tokenid'] * 1000000 +
                                                                     test_case_vent_token_series1['tokenid'])],
                                                             'LABEL': [test_case_vent_token_series2['labelLH'] + '->' +
                                                                       test_case_vent_token_series1['labelLH']],
                                                             'segend': [test_case_vent_token_series1['segend']]})
                        test_case_vent_token['time'] = timedelta_to_hour(
                            pd.to_datetime(test_case_vent_token['segend'].values[0])
                            - pd.to_datetime(test_case_vent_token['CHARTTIME'].values[0]))
                        test_case_vent_token_list.append(test_case_vent_token)
        test_case_vent_token_input = pd.concat([obj for obj in test_case_vent_token_list])
        print(len(test_case_vent_token_input))
        test_case_vent_token_input = test_case_vent_token_input.rename(
            columns={"icustay_id": "PatientID", "tokenid": "token_id", "LABEL": "token"})
        print(len(test_case_vent_token_input['token_id'].unique()))

        # form control vent tokens
        # test set
        print("dealing with test control vent")
        uniperIDs = test_control_vent['seg_id'].unique()
        uniperIDs.sort()
        datahadmID = []
        test_control_vent_token_list = []
        for id in range(len(uniperIDs)):
            datahadmID.append(test_control_vent[test_control_vent['seg_id'] == uniperIDs[id]].reset_index(drop=True))
        for i in range(len(datahadmID)):
            datahadmID[i] = datahadmID[i][(datahadmID[i]['CHARTTIME'] <= datahadmID[i]['segend'].values[0]) & (
                    datahadmID[i]['CHARTTIME'] >= datahadmID[i]['starttime'].values[0])]
            if len(datahadmID[i]) > 0:
                datahadmID[i] = datahadmID[i].sort_values(by=['CHARTTIME'])
                for j in datahadmID[i]['ITEMID'].unique():
                    temp_onevent = datahadmID[i][datahadmID[i]['ITEMID'] == j].reset_index(drop=True)
                    if len(temp_onevent) >= 2:
                        test_control_vent_token_series1 = temp_onevent[
                            ['seg_id', 'ITEMID', 'CHARTTIME', 'tokenid', 'labelLH',
                             'segend']].iloc[-1]
                        test_control_vent_token_series2 = \
                            temp_onevent[
                                ['seg_id', 'ITEMID', 'CHARTTIME', 'tokenid', 'labelLH',
                                 'segend']].iloc[-2]
                        test_control_vent_token = pd.DataFrame({'seg_id': [test_control_vent_token_series1['seg_id']],
                                                                'ITEMID': [
                                                                    test_control_vent_token_series1['ITEMID']],
                                                                'CHARTTIME': [
                                                                    test_control_vent_token_series1['CHARTTIME']],
                                                                'tokenid': [
                                                                    round(
                                                                        test_control_vent_token_series2[
                                                                            'tokenid'] * 1000000 +
                                                                        test_control_vent_token_series1['tokenid'])],
                                                                'LABEL': [
                                                                    test_control_vent_token_series2['labelLH'] + '->' +
                                                                    test_control_vent_token_series1['labelLH']],
                                                                'segend': [test_control_vent_token_series1['segend']]})
                        test_control_vent_token['time'] = timedelta_to_hour(
                            pd.to_datetime(test_control_vent_token['segend'].values[0])
                            - pd.to_datetime(test_control_vent_token['CHARTTIME'].values[0]))
                        test_control_vent_token_list.append(test_control_vent_token)
        test_control_vent_token_input = pd.concat([obj for obj in test_control_vent_token_list])
        print(len(test_control_vent_token_input))
        test_control_vent_token_input = test_control_vent_token_input.rename(
            columns={"seg_id": "PatientID", "tokenid": "token_id", "LABEL": "token"})
        print(len(test_control_vent_token_input['token_id'].unique()))
    elif vent_token_method == 'all':
        # form normal vent tokens
        # form case lab tokens
        # test set
        print("dealing with test case vent")
        uniperIDs = test_case_vent['icustay_id'].unique()
        uniperIDs.sort()
        datahadmID = []
        test_case_vent_token_list = []
        for id in range(len(uniperIDs)):
            datahadmID.append(test_case_vent[test_case_vent['icustay_id'] == uniperIDs[id]])
        for i in range(len(datahadmID)):
            datahadmID[i] = datahadmID[i][(datahadmID[i]['CHARTTIME'] <= datahadmID[i]['segend'].values[0]) & (
                    datahadmID[i]['CHARTTIME'] >= datahadmID[i]['segstart'].values[0])]
            if len(datahadmID[i]) > 0:
                datahadmID[i] = datahadmID[i].sort_values(by=['CHARTTIME'])
                for j in datahadmID[i]['tokenid'].unique():
                    temp_onevent = datahadmID[i][datahadmID[i]['tokenid'] == j]
                    if len(temp_onevent) != 0:
                        test_case_vent_token_series = \
                            temp_onevent[['icustay_id', 'CHARTTIME', 'tokenid', 'labelLH', 'segend']].iloc[-1]
                        test_case_vent_token = pd.DataFrame({'icustay_id': [test_case_vent_token_series['icustay_id']],
                                                             'CHARTTIME': [test_case_vent_token_series['CHARTTIME']],
                                                             'tokenid': [round(test_case_vent_token_series['tokenid'])],
                                                             'LABEL': [test_case_vent_token_series['labelLH']],
                                                             'segend': [test_case_vent_token_series['segend']]})
                        test_case_vent_token['time'] = timedelta_to_hour(
                            pd.to_datetime(test_case_vent_token['segend'].values[0])
                            - pd.to_datetime(
                                test_case_vent_token['CHARTTIME'].values[0]))
                        test_case_vent_token_list.append(test_case_vent_token)
        test_case_vent_token_input = pd.concat([obj for obj in test_case_vent_token_list])
        print(len(test_case_vent_token_input))
        test_case_vent_token_input = test_case_vent_token_input.rename(
            columns={"icustay_id": "PatientID", "tokenid": "token_id", "LABEL": "token"})
        print(len(test_case_vent_token_input['token_id'].unique()))

        # form control vent tokens
        # test set
        print("dealing with test control vent")
        uniperIDs = test_control_vent['seg_id'].unique()
        uniperIDs.sort()
        datahadmID = []
        test_control_vent_token_list = []
        for id in range(len(uniperIDs)):
            datahadmID.append(test_control_vent[test_control_vent['seg_id'] == uniperIDs[id]])
        for i in range(len(datahadmID)):
            datahadmID[i] = datahadmID[i][(datahadmID[i]['CHARTTIME'] <= datahadmID[i]['segend'].values[0]) & (
                    datahadmID[i]['CHARTTIME'] >= datahadmID[i]['segstart'].values[0])]
            if len(datahadmID[i]) > 0:
                datahadmID[i] = datahadmID[i].sort_values(by=['CHARTTIME'])
                for j in datahadmID[i]['tokenid'].unique():
                    temp_onevent = datahadmID[i][datahadmID[i]['tokenid'] == j]
                    if len(temp_onevent) != 0:
                        test_control_vent_token_series = \
                            temp_onevent[['seg_id', 'CHARTTIME', 'tokenid', 'labelLH', 'segend']].iloc[-1]
                        test_control_vent_token = pd.DataFrame({'seg_id': [test_control_vent_token_series['seg_id']],
                                                                'CHARTTIME': [
                                                                    test_control_vent_token_series['CHARTTIME']],
                                                                'tokenid': [
                                                                    round(test_control_vent_token_series['tokenid'])],
                                                                'LABEL': [test_control_vent_token_series['labelLH']],
                                                                'segend': [test_control_vent_token_series['segend']]})
                        test_control_vent_token['time'] = timedelta_to_hour(
                            pd.to_datetime(test_control_vent_token['segend'].values[0])
                            - pd.to_datetime(test_control_vent_token['CHARTTIME'].values[0]))
                        test_control_vent_token_list.append(test_control_vent_token)
        test_control_vent_token_input = pd.concat([obj for obj in test_control_vent_token_list])
        print(len(test_control_vent_token_input))
        test_control_vent_token_input = test_control_vent_token_input.rename(
            columns={"seg_id": "PatientID", "tokenid": "token_id", "LABEL": "token"})
        print(len(test_control_vent_token_input['token_id'].unique()))
    elif vent_token_method == 'intub_duration':
        # form dutration tokens
        # form case vent tokens
        # test set
        print("dealing with test case vent")
        uniperIDs = test_case_vent['icustay_id'].unique()
        uniperIDs.sort()
        test_case_vent_token_list = []
        for id in range(len(uniperIDs)):
            temp_onepatient = test_case_vent[test_case_vent['icustay_id'] == uniperIDs[id]].iloc[-1]
            intub_duration = round(timedelta_to_hour(pd.to_datetime(temp_onepatient['segend'])
                                                     - pd.to_datetime(temp_onepatient['starttime'])) / 24)
            if intub_duration >= 20:
                intub_duration = 20
            test_case_vent_token = pd.DataFrame({'icustay_id': [temp_onepatient['icustay_id']],
                                                 'CHARTTIME': [temp_onepatient['segend']],
                                                 'tokenid': [round(21 * 10000 + intub_duration)],
                                                 'LABEL': ['Intubation_time_varing_duration_' + str(intub_duration)],
                                                 'segend': [temp_onepatient['segend']]})

            test_case_vent_token['time'] = 0
            test_case_vent_token_list.append(test_case_vent_token)
        test_case_vent_token_input = pd.concat([obj for obj in test_case_vent_token_list])
        print(len(test_case_vent_token_input))
        test_case_vent_token_input = test_case_vent_token_input.rename(
            columns={"icustay_id": "PatientID", "tokenid": "token_id", "LABEL": "token"})
        print(len(test_case_vent_token_input['token_id'].unique()))

        # form control vent tokens
        # test set
        print("dealing with test control vent")
        uniperIDs = test_control_vent['seg_id'].unique()
        uniperIDs.sort()
        test_control_vent_token_list = []
        for id in range(len(uniperIDs)):
            temp_onepatient = test_control_vent[test_control_vent['seg_id'] == uniperIDs[id]].iloc[-1]
            intub_duration = round(timedelta_to_hour(pd.to_datetime(temp_onepatient['segend'])
                                                     - pd.to_datetime(temp_onepatient['starttime'])) / 24)
            if intub_duration >= 20:
                intub_duration = 20
            test_control_vent_token = pd.DataFrame({'seg_id': [temp_onepatient['seg_id']],
                                                    'CHARTTIME': [temp_onepatient['segend']],
                                                    'tokenid': [round(21 * 10000 + intub_duration)],
                                                    'LABEL': ['Intubation_time_varing_duration_' + str(intub_duration)],
                                                    'segend': [temp_onepatient['segend']]})
            test_control_vent_token['time'] = 0
            test_control_vent_token_list.append(test_control_vent_token)
        test_control_vent_token_input = pd.concat([obj for obj in test_control_vent_token_list])
        print(len(test_control_vent_token_input))
        test_control_vent_token_input = test_control_vent_token_input.rename(
            columns={"seg_id": "PatientID", "tokenid": "token_id", "LABEL": "token"})
        print(len(test_control_vent_token_input['token_id'].unique()))

    return test_case_vent_token_input, \
           test_control_vent_token_input


def GetDemoTokens(test_case_segs, test_control_segs, demographics):
    demographics = demographics.rename(columns={"stay_id": "icustay_id"})
    test_case = test_case_segs[['hadm_id', 'icustay_id', 'segend', 'segstart']]
    test_control = test_control_segs[['hadm_id', 'icustay_id', 'segend', 'segstart']]

    test_case_demo = pd.merge(test_case, demographics, how="left", on=["icustay_id"]).drop_duplicates(keep='first')
    test_control_demo = pd.merge(test_control, demographics, how="left", on=["icustay_id"]).drop_duplicates(
        keep='first')

    test_case_demo['segend'] = pd.to_datetime(test_case_demo['segend'])
    test_case_demo['segstart'] = pd.to_datetime(test_case_demo['segstart'])

    # form case demo tokens
    # test set
    print("dealing with test case lab")
    uniperIDs = test_case_demo['icustay_id'].unique()
    uniperIDs.sort()
    datahadmID = []
    test_case_demo_token_list = []
    for id in range(len(uniperIDs)):
        datahadmID.append(test_case_demo[test_case_demo['icustay_id'] == uniperIDs[id]])
    for i in range(len(datahadmID)):
        if len(datahadmID[i]) > 0:
            infoOfOne = datahadmID[i].iloc[-1]
            age = infoOfOne['age']
            if age >= 18 and age < 45:
                test_case_demo_token_age = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                         'CHARTTIME': [infoOfOne['segend']],
                                                         'VALUE': [infoOfOne['age']],
                                                         'ITEMID': [300010],
                                                         'LABEL': ['age18-44'],
                                                         'segend': [infoOfOne['segend']]})
            elif age >= 45 and age < 65:
                test_case_demo_token_age = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                         'CHARTTIME': [infoOfOne['segend']],
                                                         'VALUE': [infoOfOne['age']],
                                                         'ITEMID': [300011],
                                                         'LABEL': ['age45-64'],
                                                         'segend': [infoOfOne['segend']]})
            else:
                test_case_demo_token_age = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                         'CHARTTIME': [infoOfOne['segend']],
                                                         'VALUE': [infoOfOne['age']],
                                                         'ITEMID': [300012],
                                                         'LABEL': ['age65-'],
                                                         'segend': [infoOfOne['segend']]})
            test_case_demo_token_age['time'] = 0
            test_case_demo_token_list.append(test_case_demo_token_age)
            gender = datahadmID[i].iloc[-1]['gender']
            if gender == 'F':
                test_case_demo_token_gender = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                            'CHARTTIME': [infoOfOne['segend']],
                                                            'VALUE': [infoOfOne['gender']],
                                                            'ITEMID': [300020],
                                                            'LABEL': ['F'],
                                                            'segend': [infoOfOne['segend']]})
            else:
                test_case_demo_token_gender = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                            'CHARTTIME': [infoOfOne['segend']],
                                                            'VALUE': [infoOfOne['gender']],
                                                            'ITEMID': [300021],
                                                            'LABEL': ['M'],
                                                            'segend': [infoOfOne['segend']]})
            test_case_demo_token_gender['time'] = 0
            test_case_demo_token_list.append(test_case_demo_token_gender)
            ethnicity = datahadmID[i].iloc[-1]['ethnicity']
            if ethnicity == 'WHITE':
                test_case_demo_token_ethnicity = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                               'CHARTTIME': [infoOfOne['segend']],
                                                               'VALUE': [infoOfOne['ethnicity']],
                                                               'ITEMID': [300030],
                                                               'LABEL': ['WHITE'],
                                                               'segend': [infoOfOne['segend']]})
            elif ethnicity == 'BLACK':
                test_case_demo_token_ethnicity = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                               'CHARTTIME': [infoOfOne['segend']],
                                                               'VALUE': [infoOfOne['ethnicity']],
                                                               'ITEMID': [300031],
                                                               'LABEL': ['BLACK'],
                                                               'segend': [infoOfOne['segend']]})
            elif ethnicity == 'ASIAN':
                test_case_demo_token_ethnicity = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                               'CHARTTIME': [infoOfOne['segend']],
                                                               'VALUE': [infoOfOne['ethnicity']],
                                                               'ITEMID': [300032],
                                                               'LABEL': ['ASIAN'],
                                                               'segend': [infoOfOne['segend']]})
            elif ethnicity == 'HISPANIC':
                test_case_demo_token_ethnicity = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                               'CHARTTIME': [infoOfOne['segend']],
                                                               'VALUE': [infoOfOne['ethnicity']],
                                                               'ITEMID': [300033],
                                                               'LABEL': ['HISPANIC'],
                                                               'segend': [infoOfOne['segend']]})
            else:
                test_case_demo_token_ethnicity = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                               'CHARTTIME': [infoOfOne['segend']],
                                                               'VALUE': [infoOfOne['ethnicity']],
                                                               'ITEMID': [300034],
                                                               'LABEL': ['OTHER'],
                                                               'segend': [infoOfOne['segend']]})
            test_case_demo_token_ethnicity['time'] = 0
            test_case_demo_token_list.append(test_case_demo_token_ethnicity)
            height = infoOfOne['height']
            if height < 150:
                test_case_demo_token_height = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                            'CHARTTIME': [infoOfOne['segend']],
                                                            'VALUE': [infoOfOne['height']],
                                                            'ITEMID': [300040],
                                                            'LABEL': ['height<150'],
                                                            'segend': [infoOfOne['segend']]})
            elif height >= 150 and height < 160:
                test_case_demo_token_height = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                            'CHARTTIME': [infoOfOne['segend']],
                                                            'VALUE': [infoOfOne['height']],
                                                            'ITEMID': [300041],
                                                            'LABEL': ['height150-160'],
                                                            'segend': [infoOfOne['segend']]})
            elif height >= 160 and height < 170:
                test_case_demo_token_height = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                            'CHARTTIME': [infoOfOne['segend']],
                                                            'VALUE': [infoOfOne['height']],
                                                            'ITEMID': [300042],
                                                            'LABEL': ['height160-170'],
                                                            'segend': [infoOfOne['segend']]})
            elif height >= 170 and height < 180:
                test_case_demo_token_height = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                            'CHARTTIME': [infoOfOne['segend']],
                                                            'VALUE': [infoOfOne['height']],
                                                            'ITEMID': [300043],
                                                            'LABEL': ['height170-180'],
                                                            'segend': [infoOfOne['segend']]})
            elif height >= 180 and height < 190:
                test_case_demo_token_height = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                            'CHARTTIME': [infoOfOne['segend']],
                                                            'VALUE': [infoOfOne['height']],
                                                            'ITEMID': [300044],
                                                            'LABEL': ['height180-190'],
                                                            'segend': [infoOfOne['segend']]})
            else:
                test_case_demo_token_height = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                            'CHARTTIME': [infoOfOne['segend']],
                                                            'VALUE': [infoOfOne['height']],
                                                            'ITEMID': [300045],
                                                            'LABEL': ['height190+'],
                                                            'segend': [infoOfOne['segend']]})
            test_case_demo_token_height['time'] = 0
            test_case_demo_token_list.append(test_case_demo_token_height)
            BMI = infoOfOne['BMI']
            if BMI < 18.5:
                test_case_demo_token_BMI = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                         'CHARTTIME': [infoOfOne['segend']],
                                                         'VALUE': [infoOfOne['BMI']],
                                                         'ITEMID': [300050],
                                                         'LABEL': ['BMI<18.5'],
                                                         'segend': [infoOfOne['segend']]})
            elif BMI >= 18.5 and BMI < 25:
                test_case_demo_token_BMI = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                         'CHARTTIME': [infoOfOne['segend']],
                                                         'VALUE': [infoOfOne['BMI']],
                                                         'ITEMID': [300051],
                                                         'LABEL': ['BMI18.5-25'],
                                                         'segend': [infoOfOne['segend']]})
            elif BMI >= 25 and BMI < 30:
                test_case_demo_token_BMI = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                         'CHARTTIME': [infoOfOne['segend']],
                                                         'VALUE': [infoOfOne['BMI']],
                                                         'ITEMID': [300052],
                                                         'LABEL': ['BMI25-30'],
                                                         'segend': [infoOfOne['segend']]})
            elif BMI >= 30 and BMI < 35:
                test_case_demo_token_BMI = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                         'CHARTTIME': [infoOfOne['segend']],
                                                         'VALUE': [infoOfOne['BMI']],
                                                         'ITEMID': [300053],
                                                         'LABEL': ['BMI30-35'],
                                                         'segend': [infoOfOne['segend']]})
            else:
                test_case_demo_token_BMI = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                         'CHARTTIME': [infoOfOne['segend']],
                                                         'VALUE': [infoOfOne['BMI']],
                                                         'ITEMID': [300054],
                                                         'LABEL': ['BMI35+'],
                                                         'segend': [infoOfOne['segend']]})
            test_case_demo_token_BMI['time'] = 0
            test_case_demo_token_list.append(test_case_demo_token_BMI)
    test_case_demo_token_input = pd.concat([obj for obj in test_case_demo_token_list])
    print(len(test_case_demo_token_input))
    test_case_lab_token_input = test_case_demo_token_input.rename(
        columns={"icustay_id": "PatientID", "VALUE": "value", "ITEMID": "token_id", "LABEL": "token"})
    print(len(test_case_lab_token_input['token_id'].unique()))

    # form control demo tokens
    # test set
    print("dealing with test control lab")
    uniperIDs = test_control_demo['icustay_id'].unique()
    uniperIDs.sort()
    datahadmID = []
    test_control_demo_token_list = []
    for id in range(len(uniperIDs)):
        datahadmID.append(test_control_demo[test_control_demo['icustay_id'] == uniperIDs[id]])
    for i in range(len(datahadmID)):
        if len(datahadmID[i]) > 0:
            infoOfOne = datahadmID[i].iloc[-1]
            age = infoOfOne['age']
            if age >= 18 and age < 45:
                test_control_demo_token_age = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                            'CHARTTIME': [infoOfOne['segend']],
                                                            'VALUE': [infoOfOne['age']],
                                                            'ITEMID': [300010],
                                                            'LABEL': ['age18-44'],
                                                            'segend': [infoOfOne['segend']]})
            elif age >= 45 and age < 65:
                test_control_demo_token_age = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                            'CHARTTIME': [infoOfOne['segend']],
                                                            'VALUE': [infoOfOne['age']],
                                                            'ITEMID': [300011],
                                                            'LABEL': ['age45-64'],
                                                            'segend': [infoOfOne['segend']]})
            else:
                test_control_demo_token_age = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                            'CHARTTIME': [infoOfOne['segend']],
                                                            'VALUE': [infoOfOne['age']],
                                                            'ITEMID': [300012],
                                                            'LABEL': ['age65-'],
                                                            'segend': [infoOfOne['segend']]})
            test_control_demo_token_age['time'] = 0
            test_control_demo_token_list.append(test_control_demo_token_age)
            gender = datahadmID[i].iloc[-1]['gender']
            if gender == 'F':
                test_control_demo_token_gender = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                               'CHARTTIME': [infoOfOne['segend']],
                                                               'VALUE': [infoOfOne['gender']],
                                                               'ITEMID': [300020],
                                                               'LABEL': ['F'],
                                                               'segend': [infoOfOne['segend']]})
            else:
                test_control_demo_token_gender = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                               'CHARTTIME': [infoOfOne['segend']],
                                                               'VALUE': [infoOfOne['gender']],
                                                               'ITEMID': [300021],
                                                               'LABEL': ['M'],
                                                               'segend': [infoOfOne['segend']]})
            test_control_demo_token_gender['time'] = 0
            test_control_demo_token_list.append(test_control_demo_token_gender)
            ethnicity = datahadmID[i].iloc[-1]['ethnicity']
            if ethnicity == 'WHITE':
                test_control_demo_token_ethnicity = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                                  'CHARTTIME': [infoOfOne['segend']],
                                                                  'VALUE': [infoOfOne['ethnicity']],
                                                                  'ITEMID': [300030],
                                                                  'LABEL': ['WHITE'],
                                                                  'segend': [infoOfOne['segend']]})
            elif ethnicity == 'BLACK':
                test_control_demo_token_ethnicity = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                                  'CHARTTIME': [infoOfOne['segend']],
                                                                  'VALUE': [infoOfOne['ethnicity']],
                                                                  'ITEMID': [300031],
                                                                  'LABEL': ['BLACK'],
                                                                  'segend': [infoOfOne['segend']]})
            elif ethnicity == 'ASIAN':
                test_control_demo_token_ethnicity = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                                  'CHARTTIME': [infoOfOne['segend']],
                                                                  'VALUE': [infoOfOne['ethnicity']],
                                                                  'ITEMID': [300032],
                                                                  'LABEL': ['ASIAN'],
                                                                  'segend': [infoOfOne['segend']]})
            elif ethnicity == 'HISPANIC':
                test_control_demo_token_ethnicity = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                                  'CHARTTIME': [infoOfOne['segend']],
                                                                  'VALUE': [infoOfOne['ethnicity']],
                                                                  'ITEMID': [300033],
                                                                  'LABEL': ['HISPANIC'],
                                                                  'segend': [infoOfOne['segend']]})
            else:
                test_control_demo_token_ethnicity = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                                  'CHARTTIME': [infoOfOne['segend']],
                                                                  'VALUE': [infoOfOne['ethnicity']],
                                                                  'ITEMID': [300034],
                                                                  'LABEL': ['OTHER'],
                                                                  'segend': [infoOfOne['segend']]})
            test_control_demo_token_ethnicity['time'] = 0
            test_control_demo_token_list.append(test_control_demo_token_ethnicity)
            height = infoOfOne['height']
            if height < 150:
                test_control_demo_token_height = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                               'CHARTTIME': [infoOfOne['segend']],
                                                               'VALUE': [infoOfOne['height']],
                                                               'ITEMID': [300040],
                                                               'LABEL': ['height<150'],
                                                               'segend': [infoOfOne['segend']]})
            elif height >= 150 and height < 160:
                test_control_demo_token_height = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                               'CHARTTIME': [infoOfOne['segend']],
                                                               'VALUE': [infoOfOne['height']],
                                                               'ITEMID': [300041],
                                                               'LABEL': ['height150-160'],
                                                               'segend': [infoOfOne['segend']]})
            elif height >= 160 and height < 170:
                test_control_demo_token_height = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                               'CHARTTIME': [infoOfOne['segend']],
                                                               'VALUE': [infoOfOne['height']],
                                                               'ITEMID': [300042],
                                                               'LABEL': ['height160-170'],
                                                               'segend': [infoOfOne['segend']]})
            elif height >= 170 and height < 180:
                test_control_demo_token_height = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                               'CHARTTIME': [infoOfOne['segend']],
                                                               'VALUE': [infoOfOne['height']],
                                                               'ITEMID': [300043],
                                                               'LABEL': ['height170-180'],
                                                               'segend': [infoOfOne['segend']]})
            elif height >= 180 and height < 190:
                test_control_demo_token_height = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                               'CHARTTIME': [infoOfOne['segend']],
                                                               'VALUE': [infoOfOne['height']],
                                                               'ITEMID': [300044],
                                                               'LABEL': ['height180-190'],
                                                               'segend': [infoOfOne['segend']]})
            else:
                test_control_demo_token_height = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                               'CHARTTIME': [infoOfOne['segend']],
                                                               'VALUE': [infoOfOne['height']],
                                                               'ITEMID': [300045],
                                                               'LABEL': ['height190+'],
                                                               'segend': [infoOfOne['segend']]})
            test_control_demo_token_height['time'] = 0
            test_control_demo_token_list.append(test_control_demo_token_height)
            BMI = infoOfOne['BMI']
            if BMI < 18.5:
                test_control_demo_token_BMI = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                            'CHARTTIME': [infoOfOne['segend']],
                                                            'VALUE': [infoOfOne['BMI']],
                                                            'ITEMID': [300050],
                                                            'LABEL': ['BMI<18.5'],
                                                            'segend': [infoOfOne['segend']]})
            elif BMI >= 18.5 and BMI < 25:
                test_control_demo_token_BMI = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                            'CHARTTIME': [infoOfOne['segend']],
                                                            'VALUE': [infoOfOne['BMI']],
                                                            'ITEMID': [300051],
                                                            'LABEL': ['BMI18.5-25'],
                                                            'segend': [infoOfOne['segend']]})
            elif BMI >= 25 and BMI < 30:
                test_control_demo_token_BMI = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                            'CHARTTIME': [infoOfOne['segend']],
                                                            'VALUE': [infoOfOne['BMI']],
                                                            'ITEMID': [300052],
                                                            'LABEL': ['BMI25-30'],
                                                            'segend': [infoOfOne['segend']]})
            elif BMI >= 30 and BMI < 35:
                test_control_demo_token_BMI = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                            'CHARTTIME': [infoOfOne['segend']],
                                                            'VALUE': [infoOfOne['BMI']],
                                                            'ITEMID': [300053],
                                                            'LABEL': ['BMI30-35'],
                                                            'segend': [infoOfOne['segend']]})
            else:
                test_control_demo_token_BMI = pd.DataFrame({'icustay_id': [infoOfOne['icustay_id']],
                                                            'CHARTTIME': [infoOfOne['segend']],
                                                            'VALUE': [infoOfOne['BMI']],
                                                            'ITEMID': [300054],
                                                            'LABEL': ['BMI35+'],
                                                            'segend': [infoOfOne['segend']]})
            test_control_demo_token_BMI['time'] = 0
            test_control_demo_token_list.append(test_control_demo_token_BMI)
    test_control_demo_token_input = pd.concat([obj for obj in test_control_demo_token_list])
    print(len(test_control_demo_token_input))
    test_control_lab_token_input = test_control_demo_token_input.rename(
        columns={"icustay_id": "PatientID", "VALUE": "value", "ITEMID": "token_id", "LABEL": "token"})
    print(len(test_control_lab_token_input['token_id'].unique()))
    return test_case_lab_token_input, test_control_lab_token_input


def formattedVitalTokenInput(file_path):
    vital_list = ['heartrate', 'sysbp', 'meanbp', 'spo2', 'tempc', 'resprate']
    top_feature_path = file_path.replace('mimiciv/', '')
    # form final vital inputs
    # generate token and timepoints from feature tables, set onset as time 0
    # from token inputs for the whole training set
    vital_digit = 0
    test_case_vital_token = []
    test_control_vital_token = []
    for vital_name in vital_list:
        vital_digit = vital_digit + 1
        vital_top40 = pd.read_csv(
            top_feature_path + 'l1trend_features/top_features/' + vital_name + '_TOP40.csv')
        for i in range(len(vital_top40)):
            delta = format(vital_top40.at[i, 'delta'], '.1g')
            predictor = vital_top40.at[i, 'predictors']
            value = vital_top40.at[i, 'value']
            label = vital_top40.at[i, 'label']
            test_case_feature = \
                pd.read_csv(
                    file_path + 'l1trend_features/l1trendfeature_' + delta + '_' + vital_name + '_test_case.csv')[
                    ['icustay_id', vital_top40.at[i, 'predictors']]]
            test_control_feature = \
                pd.read_csv(
                    file_path + 'l1trend_features/l1trendfeature_' + delta + '_' + vital_name + '_test_control.csv')[
                    ['seg_id', vital_top40.at[i, 'predictors']]]
            if label == 1:
                test_case_token = test_case_feature[test_case_feature[predictor] >= value]
                test_case_token['time'] = 0
                test_case_token['token'] = predictor + '_' + str(label)
                test_case_token['token_id'] = 10000 + vital_digit * 1000 + i + 1
                test_case_token = test_case_token.rename(columns={"icustay_id": "PatientID", predictor: "value"})
                test_control_token = test_control_feature[test_control_feature[predictor] >= value]
                test_control_token['time'] = 0
                test_control_token['token'] = predictor + '_' + str(label)
                test_control_token['token_id'] = 10000 + vital_digit * 1000 + i + 1
                test_control_token = test_control_token.rename(columns={"seg_id": "PatientID", predictor: "value"})
            else:
                test_case_token = test_case_feature.loc[test_case_feature[predictor] <= value]
                test_case_token['time'] = 0
                test_case_token['token'] = predictor + '_' + str(label)
                test_case_token['token_id'] = 10000 + vital_digit * 1000 + i + 1
                test_case_token = test_case_token.rename(columns={"icustay_id": "PatientID", predictor: "value"})
                test_control_token = test_control_feature[test_control_feature[predictor] <= value]
                test_control_token['time'] = 0
                test_control_token['token'] = predictor + '_' + str(label)
                test_control_token['token_id'] = 10000 + vital_digit * 1000 + i + 1
                test_control_token = test_control_token.rename(columns={"seg_id": "PatientID", predictor: "value"})
            test_case_vital_token.append(test_case_token)
            test_control_vital_token.append(test_control_token)
    folder_path = file_path + 'tokens/vital'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    test_case_vital_token_input = pd.concat([obj for obj in test_case_vital_token])
    test_control_vital_token_input = pd.concat([obj for obj in test_control_vital_token])
    test_case_vital_token_input.to_csv(
        folder_path + '/test_case_vital_token_input.csv')
    test_control_vital_token_input.to_csv(
        folder_path + '/test_control_vital_token_input.csv')




