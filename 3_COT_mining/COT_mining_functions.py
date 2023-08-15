import pandas as pd
import numpy as np
import os
import math
import time
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
from sklearn.model_selection import KFold
import ast

mafia_path = ''

def TokenInputToBinaryCodeMatrix(
        dataset_patientid, vital_token, lab_token, vent_all_token, vent_lasttwo_token, vent_intub_duration_token,
        demo_token, tokenid_num):
    uniperIDs = dataset_patientid['PatientID'].unique()
    uniperIDs.sort()
    all_token = pd.concat([val[['PatientID', 'token', 'token_id']]
                           for val in [vital_token, lab_token, vent_all_token,
                                       vent_lasttwo_token, vent_intub_duration_token,
                                       demo_token] if len(val) != 0])
    code_matrix = np.zeros((len(uniperIDs), len(tokenid_num)))
    for id in range(len(uniperIDs)):
        person_token = all_token[all_token['PatientID'] == uniperIDs[id]]
        if len(person_token) > 0:
            for tokenid in person_token['token_id']:
                if len(tokenid_num[tokenid_num['token_id'] == tokenid]) != 0:
                    # in case that the tokenid is not in the tokenid_num
                    code_matrix[id, tokenid_num['token_loc'][tokenid_num['token_id'] == tokenid].values[0]] = 1
    print(code_matrix.shape)
    return code_matrix


def GenerateCodeMatrix4CV(generate_path):
    # get train segments
    fold_num = 5
    train_case_seg = pd.read_csv(generate_path + 'segments/train_case_segs.csv')
    train_case_seg['PatientID'] = train_case_seg['icustay_id']
    train_control_seg = pd.read_csv(generate_path + 'segments/train_control_segs.csv')
    train_control_seg['PatientID'] = train_control_seg['seg_id']

    # get all the maps
    vital_map = pd.read_csv(generate_path + 'tokens/map/vital_map.csv')
    vent_lasttwo_map = pd.read_csv(generate_path + 'tokens/map/vent_map_lasttwo.csv')
    vent_all_map = pd.read_csv(generate_path + 'tokens/map/vent_map_all.csv')
    vent_intub_duration_map = pd.read_csv(generate_path + 'tokens/map/vent_map_intub_duration.csv')
    demo_map = pd.read_csv(generate_path + 'tokens/map/demo_map.csv')
    lab_map = pd.read_csv(generate_path + 'tokens/map/lab_map.csv')

    # get all the token inputs
    train_case_lab_token_input = pd.read_csv(generate_path + 'tokens/lab/train_case_lab_token_input.csv')
    train_control_lab_token_input = pd.read_csv(generate_path +
                                                'tokens/lab/train_control_lab_token_input.csv')
    train_case_vent_all_token_input = pd.read_csv(
        generate_path + 'tokens/vent/train_case_vent_all_token_input.csv')
    train_case_vent_intub_duration_token_input = pd.read_csv(
        generate_path + 'tokens/vent/train_case_vent_intub_duration_token_input.csv')
    train_case_vent_lasttwo_token_input = pd.read_csv(
        generate_path + 'tokens/vent/train_case_vent_lasttwo_token_input.csv')
    train_control_vent_all_token_input = pd.read_csv(
        generate_path + 'tokens/vent/train_control_vent_all_token_input.csv')
    train_control_vent_intub_duration_token_input = pd.read_csv(
        generate_path + 'tokens/vent/train_control_vent_intub_duration_token_input.csv')
    train_control_vent_lasttwo_token_input = pd.read_csv(
        generate_path + 'tokens/vent/train_control_vent_lasttwo_token_input.csv')
    train_case_demo_token_input = pd.read_csv(generate_path + 'tokens/demo/train_case_demo_token_input.csv')
    train_control_demo_token_input = pd.read_csv(generate_path + 'tokens/demo/train_control_demo_token_input.csv')
    train_case_vital_token_input = pd.read_csv(generate_path + 'tokens/vital/train_case_vital_token_input.csv')
    train_control_vital_token_input = pd.read_csv(generate_path + 'tokens/vital/train_control_vital_token_input.csv')

    print("CV data...")
    kf = KFold(n_splits=fold_num)
    train_case_CVtrainindex_list = []
    train_case_CVvalindex_list = []
    for train_index, val_index in kf.split(train_case_seg):
        train_case_CVtrainindex_list.append(train_index)
        train_case_CVvalindex_list.append(val_index)

    # control is divided by subject_id
    train_control_CVtrainindex_list = []
    train_control_CVvalindex_list = []
    print('control_whole_train:', len(train_control_seg))
    print('control_whole_train subject:', len(train_control_seg['subject_id'].unique()))
    for train_index, val_index in kf.split(train_control_seg['subject_id'].unique()):
        non_train_subid = train_control_seg['subject_id'].unique()[train_index]
        non_val_subid = train_control_seg['subject_id'].unique()[val_index]
        train_control_CVtrainindex_list.append(
            train_control_seg[train_control_seg['subject_id'].isin(non_train_subid)].index)
        train_control_CVvalindex_list.append(
            train_control_seg[train_control_seg['subject_id'].isin(non_val_subid)].index)

    tokenid_num = pd.concat([val for val in [vital_map, lab_map, vent_all_map, vent_lasttwo_map,
                                             vent_intub_duration_map, demo_map] if len(val) != 0])
    tokenid_num['token_loc'] = list(range(len(tokenid_num)))  # remember the location starts from 0
    folder_path = generate_path + 'tokens/matrix'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    tokenid_num.to_csv(
        folder_path + '/tokenid_loc.csv')  # [['token_id', 'token'， 'token_loc']]
    print(tokenid_num.head())

    for indexToFold in range(fold_num):
        print("CV:", indexToFold)
        train_case_cv, val_case_cv = train_case_seg[['PatientID']].iloc[train_case_CVtrainindex_list[indexToFold]], \
            train_case_seg[['PatientID']].iloc[train_case_CVvalindex_list[indexToFold]]
        train_case_vital_token_input_cv = train_case_vital_token_input[train_case_vital_token_input['PatientID'].isin(
            train_case_cv['PatientID'])]
        val_case_vital_token_input_cv = train_case_vital_token_input[train_case_vital_token_input['PatientID'].isin(
            val_case_cv['PatientID'])]
        train_case_lab_token_input_cv = train_case_lab_token_input[train_case_lab_token_input['PatientID'].isin(
            train_case_cv['PatientID'])]
        val_case_lab_token_input_cv = train_case_lab_token_input[train_case_lab_token_input['PatientID'].isin(
            val_case_cv['PatientID'])]
        train_case_vent_all_token_input_cv = train_case_vent_all_token_input[
            train_case_vent_all_token_input['PatientID'].isin(train_case_cv['PatientID'])]
        val_case_vent_all_token_input_cv = train_case_vent_all_token_input[
            train_case_vent_all_token_input['PatientID'].isin(val_case_cv['PatientID'])]
        train_case_vent_lasttwo_token_input_cv = train_case_vent_lasttwo_token_input[
            train_case_vent_lasttwo_token_input['PatientID'].isin(train_case_cv['PatientID'])]
        val_case_vent_lasttwo_token_input_cv = train_case_vent_lasttwo_token_input[
            train_case_vent_lasttwo_token_input['PatientID'].isin(val_case_cv['PatientID'])]
        train_case_vent_intub_duration_token_input_cv = train_case_vent_intub_duration_token_input[
            train_case_vent_intub_duration_token_input['PatientID'].isin(train_case_cv['PatientID'])]
        val_case_vent_intub_duration_token_input_cv = train_case_vent_intub_duration_token_input[
            train_case_vent_intub_duration_token_input['PatientID'].isin(val_case_cv['PatientID'])]
        train_case_demo_token_input_cv = train_case_demo_token_input[
            train_case_demo_token_input['PatientID'].isin(train_case_cv['PatientID'])]
        val_case_demo_token_input_cv = train_case_demo_token_input[
            train_case_demo_token_input['PatientID'].isin(val_case_cv['PatientID'])]

        train_control_cv, val_control_cv = train_control_seg[['PatientID']].iloc[
            train_control_CVtrainindex_list[indexToFold]], \
            train_control_seg[['PatientID']].iloc[train_control_CVvalindex_list[indexToFold]]
        train_control_vital_token_input_cv = train_control_vital_token_input[
            train_control_vital_token_input['PatientID'].isin(train_control_cv['PatientID'])]
        val_control_vital_token_input_cv = train_control_vital_token_input[
            train_control_vital_token_input['PatientID'].isin(val_control_cv['PatientID'])]
        train_control_lab_token_input_cv = train_control_lab_token_input[
            train_control_lab_token_input['PatientID'].isin(train_control_cv['PatientID'])]
        val_control_lab_token_input_cv = train_control_lab_token_input[
            train_control_lab_token_input['PatientID'].isin(val_control_cv['PatientID'])]
        train_control_vent_all_token_input_cv = train_control_vent_all_token_input[
            train_control_vent_all_token_input['PatientID'].isin(train_control_cv['PatientID'])]
        val_control_vent_all_token_input_cv = train_control_vent_all_token_input[
            train_control_vent_all_token_input['PatientID'].isin(val_control_cv['PatientID'])]
        train_control_vent_lasttwo_token_input_cv = train_control_vent_lasttwo_token_input[
            train_control_vent_lasttwo_token_input['PatientID'].isin(train_control_cv['PatientID'])]
        val_control_vent_lasttwo_token_input_cv = train_control_vent_lasttwo_token_input[
            train_control_vent_lasttwo_token_input['PatientID'].isin(val_control_cv['PatientID'])]
        train_control_vent_intub_duration_token_input_cv = train_control_vent_intub_duration_token_input[
            train_control_vent_intub_duration_token_input['PatientID'].isin(train_control_cv['PatientID'])]
        val_control_vent_intub_duration_token_input_cv = train_control_vent_intub_duration_token_input[
            train_control_vent_intub_duration_token_input['PatientID'].isin(val_control_cv['PatientID'])]
        train_control_demo_token_input_cv = train_control_demo_token_input[
            train_control_demo_token_input['PatientID'].isin(train_control_cv['PatientID'])]
        val_control_demo_token_input_cv = train_control_demo_token_input[
            train_control_demo_token_input['PatientID'].isin(val_control_cv['PatientID'])]

        train_case_codematrix_cv = TokenInputToBinaryCodeMatrix(
            train_case_cv, train_case_vital_token_input_cv, train_case_lab_token_input_cv,
            train_case_vent_all_token_input_cv, train_case_vent_lasttwo_token_input,
            train_case_vent_intub_duration_token_input_cv,
            train_case_demo_token_input_cv, tokenid_num)
        cv_folder_path = generate_path + "tokens/matrix/cross_validation"
        if not os.path.exists(cv_folder_path):
            os.makedirs(cv_folder_path)
        np.savetxt(cv_folder_path + "/train_case_fold" + str(indexToFold) + "_codematrix.txt",
                   train_case_codematrix_cv, fmt='%d', delimiter=',')
        val_case_codematrix_cv = TokenInputToBinaryCodeMatrix(
            val_case_cv, val_case_vital_token_input_cv, val_case_lab_token_input_cv,
            val_case_vent_all_token_input_cv,
            val_case_vent_lasttwo_token_input_cv, val_case_vent_intub_duration_token_input_cv,
            val_case_demo_token_input_cv, tokenid_num)
        np.savetxt(cv_folder_path + "/val_case_fold" + str(indexToFold) + "_codematrix.txt",
                   val_case_codematrix_cv, fmt='%d', delimiter=',')
        train_control_codematrix_cv = TokenInputToBinaryCodeMatrix(
            train_control_cv, train_control_vital_token_input_cv, train_control_lab_token_input_cv,
            train_control_vent_all_token_input_cv,
            train_control_vent_lasttwo_token_input_cv, train_control_vent_intub_duration_token_input_cv,
            train_control_demo_token_input_cv, tokenid_num)
        np.savetxt(cv_folder_path + "/train_control_fold" + str(indexToFold) + "_codematrix.txt",
                   train_control_codematrix_cv, fmt='%d', delimiter=',')
        val_control_codematrix_cv = TokenInputToBinaryCodeMatrix(
            val_control_cv, val_control_vital_token_input_cv, val_control_lab_token_input_cv,
            val_control_vent_all_token_input_cv,
            val_control_vent_lasttwo_token_input_cv, val_control_vent_intub_duration_token_input_cv,
            val_control_demo_token_input_cv, tokenid_num)
        np.savetxt(cv_folder_path + "/val_control_fold" + str(indexToFold) + "_codematrix.txt",
                   val_control_codematrix_cv, fmt='%d', delimiter=',')


def GenerateCodeMatrix4Whole(generate_path):
    # get train segments
    train_case_seg = pd.read_csv(generate_path + 'segments/train_case_segs.csv')
    train_case_seg['PatientID'] = train_case_seg['icustay_id']
    train_case = train_case_seg[['PatientID']]
    train_control_seg = pd.read_csv(generate_path + 'segments/train_control_segs.csv')
    train_control_seg['PatientID'] = train_control_seg['seg_id']
    train_control = train_control_seg[['PatientID']]
    test_case_seg = pd.read_csv(generate_path + 'segments/test_case_segs.csv')
    test_case_seg['PatientID'] = test_case_seg['icustay_id']
    test_case = test_case_seg[['PatientID']]
    test_control_seg = pd.read_csv(generate_path + 'segments/test_control_segs.csv')
    test_control_seg['PatientID'] = test_control_seg['seg_id']
    test_control = test_control_seg[['PatientID']]

    # get all the token inputs
    train_case_vital_token_input = pd.read_csv(generate_path + 'tokens/vital/train_case_vital_token_input.csv')
    train_control_vital_token_input = pd.read_csv(generate_path +
                                                  'tokens/vital/train_control_vital_token_input.csv')
    train_case_lab_token_input = pd.read_csv(generate_path + 'tokens/lab/train_case_lab_token_input.csv')
    train_control_lab_token_input = pd.read_csv(generate_path +
                                                'tokens/lab/train_control_lab_token_input.csv')
    train_case_vent_all_token_input = pd.read_csv(
        generate_path + 'tokens/vent/train_case_vent_all_token_input.csv')
    train_case_vent_intub_duration_token_input = pd.read_csv(
        generate_path + 'tokens/vent/train_case_vent_intub_duration_token_input.csv')
    train_case_vent_lasttwo_token_input = pd.read_csv(
        generate_path + 'tokens/vent/train_case_vent_lasttwo_token_input.csv')
    train_control_vent_all_token_input = pd.read_csv(
        generate_path + 'tokens/vent/train_control_vent_all_token_input.csv')
    train_control_vent_intub_duration_token_input = pd.read_csv(
        generate_path + 'tokens/vent/train_control_vent_intub_duration_token_input.csv')
    train_control_vent_lasttwo_token_input = pd.read_csv(
        generate_path + 'tokens/vent/train_control_vent_lasttwo_token_input.csv')
    train_case_demo_token_input = pd.read_csv(generate_path + 'tokens/demo/train_case_demo_token_input.csv')
    train_control_demo_token_input = pd.read_csv(generate_path + 'tokens/demo/train_control_demo_token_input.csv')
    test_case_vital_token_input = pd.read_csv(generate_path + 'tokens/vital/test_case_vital_token_input.csv')
    test_control_vital_token_input = pd.read_csv(generate_path + 'tokens/vital/test_control_vital_token_input.csv')
    test_case_lab_token_input = pd.read_csv(generate_path + 'tokens/lab/test_case_lab_token_input.csv')
    test_control_lab_token_input = pd.read_csv(
        generate_path + 'tokens/lab/test_control_lab_token_input.csv')
    test_case_vent_all_token_input = pd.read_csv(
        generate_path + 'tokens/vent/test_case_vent_all_token_input.csv')
    test_case_vent_intub_duration_token_input = pd.read_csv(
        generate_path + 'tokens/vent/test_case_vent_intub_duration_token_input.csv')
    test_case_vent_lasttwo_token_input = pd.read_csv(
        generate_path + 'tokens/vent/test_case_vent_lasttwo_token_input.csv')
    test_control_vent_all_token_input = pd.read_csv(
        generate_path + 'tokens/vent/test_control_vent_all_token_input.csv')
    test_control_vent_intub_duration_token_input = pd.read_csv(
        generate_path + 'tokens/vent/test_control_vent_intub_duration_token_input.csv')
    test_control_vent_lasttwo_token_input = pd.read_csv(
        generate_path + 'tokens/vent/test_control_vent_lasttwo_token_input.csv')
    test_control_demo_token_input = pd.read_csv(generate_path + 'tokens/demo/test_control_demo_token_input.csv')
    test_case_demo_token_input = pd.read_csv(generate_path + 'tokens/demo/test_case_demo_token_input.csv')

    tokenid_num = pd.read_csv(generate_path + 'tokens/matrix/tokenid_loc.csv')
    folder_path = generate_path + 'tokens/matrix'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    train_case_codematrix = TokenInputToBinaryCodeMatrix(
        train_case, train_case_vital_token_input, train_case_lab_token_input, train_case_vent_all_token_input,
        train_case_vent_lasttwo_token_input, train_case_vent_intub_duration_token_input,
        train_case_demo_token_input, tokenid_num)
    np.savetxt(folder_path + "/train_case_codematrix.txt", train_case_codematrix, fmt='%d',
               delimiter=',')
    train_control_codematrix = TokenInputToBinaryCodeMatrix(
        train_control, train_control_vital_token_input, train_control_lab_token_input,
        train_control_vent_all_token_input, train_control_vent_lasttwo_token_input,
        train_control_vent_intub_duration_token_input, train_control_demo_token_input, tokenid_num)
    np.savetxt(folder_path + "/train_control_codematrix.txt",
               train_control_codematrix, fmt='%d', delimiter=',')
    test_case_codematrix = TokenInputToBinaryCodeMatrix(
        test_case, test_case_vital_token_input, test_case_lab_token_input, test_case_vent_all_token_input,
        test_case_vent_lasttwo_token_input, test_case_vent_intub_duration_token_input,
        test_case_demo_token_input, tokenid_num)
    np.savetxt(folder_path + "/test_case_codematrix.txt", test_case_codematrix, fmt='%d',
               delimiter=',')
    test_control_codematrix = TokenInputToBinaryCodeMatrix(
        test_control, test_control_vital_token_input, test_control_lab_token_input,
        test_control_vent_all_token_input, test_control_vent_lasttwo_token_input,
        test_control_vent_intub_duration_token_input, test_control_demo_token_input, tokenid_num)
    np.savetxt(folder_path + "/test_control_codematrix.txt", test_control_codematrix, fmt='%d',
               delimiter=',')


def transfer_BinaryCMTolocCM(binaryCM):
    # transform binary to tokenloc
    # here token loc starts from 1
    locCM = []
    for i in range(len(binaryCM)):
        loc = np.array(list(range(len(binaryCM[i])))) + 1
        temp = binaryCM[i] * loc
        temp = np.array([val for val in temp if val != 0])
        if len(temp) != 0:
            locCM.append(temp)
    #    transItems = find(codeMatix(mm,:)~ = 0);#这里可以用相乘找到loc(loc从1开始)
    return locCM


def GetPatternFP(train_control_AlarmSet, train_control_PatientIDSet, train_control_patID):
    FPSET = {}
    FPlist = []
    Patternlist = []
    num_patient = len(train_control_patID)
    for patternlen in train_control_AlarmSet.keys():
        pattern_FP = {}
        for i in range(len(train_control_AlarmSet[patternlen])):
            fp_temp = len(train_control_PatientIDSet[patternlen][i]) / num_patient
            FPlist.append(fp_temp)
            Patternlist.append(train_control_AlarmSet[patternlen][i])
            pattern_FP[i] = fp_temp
        FPSET[patternlen] = pattern_FP
    return FPSET, FPlist, Patternlist


def the_process(kk, itemsLen, jj, patID, numPat, AlarmSetMFIFromMafia, codeMatrix):
    aPatIDSet = [kk]
    vect = AlarmSetMFIFromMafia[jj][kk]  # the kk th pattern whose lenth is jj
    a = np.ones(numPat)
    for mm in range(len(vect)):
        a = a * np.array(codeMatrix[:, vect[mm] - 1])  # since token num starts from 1
    aPatIDSet.append([val for val in list(np.array(patID) * a) if val != 0])
    return aPatIDSet


def parallel_zzzGetMFIfromMAFIA(szMFITxtFile, codeMatrix, patID):
    # szMFITxtFile: szOutput4Mafia(order and minS)
    # codeMatrix: Binary matrix
    # patID: a list of patient ID
    # tic11 = time.time()
    f = open(szMFITxtFile)
    ### find the maximum length of MFI
    maxLen = -1
    szMFITxt_Matrix = []
    AlarmSetMFIFromMafia = {}
    for line2 in f:
        tempAlarmSetCode = line2.split(" ")[:-1]
        tempAlarmSetCode = list(map(int, tempAlarmSetCode))  # transfer str to int
        kk = len(tempAlarmSetCode)
        # print(kk, tempAlarmSetCode)
        if kk != 1:
            szMFITxt_Matrix.append(tempAlarmSetCode)
            if kk not in AlarmSetMFIFromMafia.keys():
                AlarmSetMFIFromMafia[kk] = [tempAlarmSetCode]
                # print(AlarmSetMFIFromMafia[kk])
            else:
                AlarmSetMFIFromMafia[kk].append(tempAlarmSetCode)
                # print(AlarmSetMFIFromMafia[kk])
            if kk > maxLen:
                maxLen = kk
    # print('maxLen: ', maxLen)
    # print('szMFITxt_Matrix: ', szMFITxt_Matrix)
    # print(time.time()-tic11,'first half of zzz')
    # tic22 = time.time()
    numPat = len(codeMatrix)  # the number of transaction
    patientIDSet = {}
    # print(len(AlarmSetMFIFromMafia.keys()),'different lenth')
    for jj in AlarmSetMFIFromMafia.keys():  # iterate patterns under different length
        itemsLen = len(AlarmSetMFIFromMafia[jj])  # there are itemsLen patterns have length jj
        aPatIDSet = {}
        # print(itemsLen)
        partial_work = partial(the_process, itemsLen=itemsLen, jj=jj, patID=patID, numPat=numPat,
                               AlarmSetMFIFromMafia=AlarmSetMFIFromMafia,
                               codeMatrix=codeMatrix)
        pool = ThreadPool()
        results = pool.map(partial_work, range(itemsLen))
        pool.close()
        pool.join()
        for kk_idx in range(itemsLen):
            aPatIDSet[results[kk_idx][0]] = results[kk_idx][1]
        patientIDSet[jj] = aPatIDSet
    return AlarmSetMFIFromMafia, patientIDSet


def GetPatternSetByFP(FPlist, Patternlist, fp):
    patternset = []
    for i in range(len(FPlist)):
        # print(FPlist[i], math.log10(FPlist[i]), fp)
        if FPlist[i] == 0:
            patternset.append(Patternlist[i])
        elif FPlist[i] <= fp:
            patternset.append(Patternlist[i])
    return patternset


def process(patternset_sublist, val_case_AlarmSet, val_case_PatientIDSet, val_control_AlarmSet,
            val_control_PatientIDSet):
    TP_count = []
    FP_count = []
    pattern_list_tohighlight = []
    TP4EachPattern = []
    for pattern in patternset_sublist:
        # print(len(pattern), ':', pattern)
        # print(val_case_AlarmSet.keys())
        if len(pattern) in val_case_AlarmSet.keys():
            if pattern in val_case_AlarmSet[len(pattern)]:
                # print('in case')
                index = val_case_AlarmSet[len(pattern)].index(pattern)
                TP_count.extend(val_case_PatientIDSet[len(pattern)][index])
                pattern_list_tohighlight.append(pattern)
                TP4EachPattern.append(len(val_case_PatientIDSet[len(pattern)][index]))
        if len(pattern) in val_control_AlarmSet.keys():
            if pattern in val_control_AlarmSet[len(pattern)]:
                # print('in control')
                index = val_control_AlarmSet[len(pattern)].index(pattern)
                FP_count.extend(val_control_PatientIDSet[len(pattern)][index])
    return TP_count, pattern_list_tohighlight, TP4EachPattern, FP_count


def GetOptimalMinSup(MinSup_list, FPR_MAX_list, MinSup_TP, MinSup_FP, MinSup_FP_thresh):
    # get the largest TPR under each FPR_MAX
    optimal_minsup_list = []
    max_TTP_list = []
    max_FFP_list = []
    final_FP_thresh_list = []
    for FPR_MAX in FPR_MAX_list:
        optimal_minsup = -1
        max_TTP = -1
        max_FFP = -1
        final_FP_thresh = np.nan
        for i in range(len(MinSup_list)):  # iterating the MINSUP
            MinSup = MinSup_list[i]
            max_TP = -1
            max_FP = -1
            max_FP_thresh = np.nan
            for j in range(len(MinSup_FP[i])):  #
                FP = MinSup_FP[i][j]
                TP = MinSup_TP[i][j]
                FP_thresh = MinSup_FP_thresh[i][j]
                if FP <= FPR_MAX:
                    if TP > max_TP:
                        max_TP = TP
                        max_FP = FP
                        max_FP_thresh = FP_thresh
            if max_TP > max_TTP:
                max_TTP = max_TP
                max_FFP = max_FP
                final_FP_thresh = max_FP_thresh
                optimal_minsup = MinSup
        optimal_minsup_list.append(optimal_minsup)
        max_TTP_list.append(max_TTP)
        max_FFP_list.append(max_FFP)
        final_FP_thresh_list.append(final_FP_thresh)

    return optimal_minsup_list, max_TTP_list, max_FFP_list, final_FP_thresh_list


def NewGetTPRPFromOnePatternset(patternset, val_case_patID, val_case_AlarmSet, val_case_PatientIDSet,
                                val_control_patID, val_control_AlarmSet, val_control_PatientIDSet):
    case_pat_num = len(val_case_patID)
    control_pat_num = len(val_control_patID)
    TP_count = []
    FP_count = []
    pattern_list_tohighlight = []
    TP4EachPattern = []

    partial_work = partial(process, val_case_AlarmSet=val_case_AlarmSet, val_case_PatientIDSet=val_case_PatientIDSet,
                           val_control_AlarmSet=val_control_AlarmSet, val_control_PatientIDSet=val_control_PatientIDSet)
    pool = ThreadPool()
    interval = round(len(patternset) / 5)
    if interval != 0:
        patternset_sublist = [patternset[i * interval:(i + 1) * interval] for i in range(4)]
        patternset_sublist.append(patternset[4 * interval:])
    else:
        patternset_sublist = [patternset]
    results = pool.map(partial_work, patternset_sublist)
    pool.close()
    pool.join()
    # print(len(patternset_sublist),len(results))
    for pi in range(len(patternset_sublist)):
        TP_count.extend(results[pi][0])
        pattern_list_tohighlight.extend(results[pi][1])
        TP4EachPattern.extend(results[pi][2])
        FP_count.extend(results[pi][3])

    TP = len(list(set(TP_count))) / case_pat_num
    FP = len(list(set(FP_count))) / control_pat_num
    return TP, FP, pattern_list_tohighlight, TP4EachPattern


def GenerateOptimalMS(generate_path, MinSup_list, FPR_MAX_list, fold_num):
    tokenid_loc = \
        pd.read_csv(generate_path + 'tokens/matrix/tokenid_loc.csv')
    print(len(tokenid_loc))

    # conduct cross validation for different minsup
    MinSup_TP = []
    MinSup_FP = []
    MinSup_FP_thresh = []
    for currMinSup in MinSup_list:

        CVFP_list = []
        Flatten_CVFP_list = []
        CVpattern_list = []
        Flatten_CVpattern_list = []
        for indexToFold in range(fold_num):
            print("CV:", indexToFold)
            train_case_binaryCM = np.loadtxt(generate_path + "tokens/matrix/cross_validation/train_case_fold"
                                             + str(indexToFold)
                                             + "_codematrix.txt", delimiter=',')
            print(train_case_binaryCM.shape)
            train_control_binaryCM = np.loadtxt(generate_path + "tokens/matrix/cross_validation/train_control_fold"
                                                + str(indexToFold)
                                                + "_codematrix.txt", delimiter=',')

            # transform binary to tokenloc
            # here token loc starts from 1
            train_case_locCM = transfer_BinaryCMTolocCM(train_case_binaryCM)
            print(len(train_case_locCM))
            train_case_locCM_output = open(generate_path + "tokens/matrix/cross_validation/train_case_locCM_fold" +
                                           str(indexToFold) + "_" + str(currMinSup) + ".txt", 'w',
                                           encoding='gbk')
            for row in train_case_locCM:
                link = ' '
                row = [str(int(x)) for x in row]
                rowtxt = link.join(row)
                train_case_locCM_output.write(rowtxt)
                train_case_locCM_output.write('\n')
            for zero_row in range(train_case_binaryCM.shape[0] - len(train_case_locCM)):
                train_case_locCM_output.write(str(100000 + zero_row))
                train_case_locCM_output.write('\n')
            train_case_locCM_output.close()

            # get the Frequent item set from MAFIA
            cv_folder_path = generate_path + "mafia_output/cross_validation"
            if not os.path.exists(cv_folder_path):
                os.makedirs(cv_folder_path)
            szOutput4Mafia = cv_folder_path + "/Output_" + str(indexToFold) + "_" \
                             + str(currMinSup) + ".txt"
            MafiaCPP = mafia_path + ' -mfi ' + str(
                currMinSup) + ' -ascii ' + \
                       generate_path + "tokens/matrix/cross_validation/train_case_locCM_fold" + str(
                indexToFold) + "_" + str(currMinSup) + ".txt" + ' ' + szOutput4Mafia
            print(MafiaCPP)
            os.system(MafiaCPP)

            # first use train_control_locCM to calculate the FPR for each pattern
            # then use different threshold to filter the candidates
            train_control_patID = [val + 1 for val in list(range(len(train_control_binaryCM)))]
            train_control_AlarmSet, train_control_PatientIDSet = parallel_zzzGetMFIfromMAFIA(
                szOutput4Mafia, train_control_binaryCM, train_control_patID)
            # print(train_control_AlarmSet)
            # print(train_control_PatientIDSet)
            FPSET, FPlist, Patternlist = GetPatternFP(train_control_AlarmSet, train_control_PatientIDSet,
                                                      train_control_patID)
            CVFP_list.append(FPlist)
            Flatten_CVFP_list = Flatten_CVFP_list + FPlist
            print(len(Flatten_CVFP_list))
            CVpattern_list.append(Patternlist)
            Flatten_CVpattern_list = Flatten_CVpattern_list + Patternlist
            # print("pattern num:", len(FPlist))
            # print(FPlist[:10])

        # use different threshold to filter different pattern sets  {threshold: [pattern, ...]}
        FP_thresh = list(set(Flatten_CVFP_list))
        FP_thresh.sort()
        # print(FPlist)
        # print(FP_thresh)
        if len(FP_thresh) > 1:  # when all the patterns have the same FP, there is no threshold range
            minFFP = min(FP_thresh)
            maxFFP = max(FP_thresh)
            if (minFFP == 0):
                FP_thresh = np.logspace(math.log10(FP_thresh[1] / 10), math.log10(maxFFP), num=20)
            else:
                FP_thresh = np.logspace(math.log10(minFFP), math.log10(maxFFP), num=20)
        else:
            if FP_thresh[0] != 0:
                FP_thresh = [math.log10(FP_thresh[0])]
        print(FP_thresh)

        # calculate the TP and FP for each pattern set, then average them, so that we can draw the ROC for this fold
        val_case_patID_list = []
        val_case_AlarmSet_list = []
        val_case_PatientIDSet_list = []
        val_control_patID_list = []
        val_control_AlarmSet_list = []
        val_control_PatientIDSet_list = []
        for indexToFold in range(fold_num):
            szOutput4Mafia = cv_folder_path + "/Output_" + str(
                indexToFold) + "_" + str(currMinSup) + ".txt"
            val_case_binaryCM = np.loadtxt(
                generate_path + "tokens/matrix/cross_validation/val_case_fold" + str(
                    indexToFold) + "_codematrix.txt", delimiter=',')
            val_control_binaryCM = np.loadtxt(
                generate_path + "tokens/matrix/cross_validation/val_control_fold" + str(
                    indexToFold) + "_codematrix.txt", delimiter=',')

            val_case_patID = [val + 1 for val in list(range(len(val_case_binaryCM)))]
            val_case_AlarmSet, val_case_PatientIDSet = parallel_zzzGetMFIfromMAFIA(szOutput4Mafia, val_case_binaryCM,
                                                                                   val_case_patID)
            val_control_patID = [val + 1 for val in list(range(len(val_control_binaryCM)))]
            val_control_AlarmSet, val_control_PatientIDSet = parallel_zzzGetMFIfromMAFIA(szOutput4Mafia,
                                                                                         val_control_binaryCM,
                                                                                         val_control_patID)
            val_case_patID_list.append(val_case_patID)
            val_case_AlarmSet_list.append(val_case_AlarmSet)
            val_case_PatientIDSet_list.append(val_case_PatientIDSet)
            val_control_patID_list.append(val_control_patID)
            val_control_AlarmSet_list.append(val_control_AlarmSet)
            val_control_PatientIDSet_list.append(val_control_PatientIDSet)

        TP_list = []
        FP_list = []
        patternset_list = []
        for fp in FP_thresh:
            # print('threshold: ', fp)
            TP_temp = 0
            FP_temp = 0
            for indexToFold in range(fold_num):
                print("CV:", indexToFold)
                patternset = GetPatternSetByFP(CVFP_list[indexToFold], CVpattern_list[indexToFold],
                                               fp)  # [PATTERN,...]
                patternset_list.append(patternset)
                if len(patternset) != 0:
                    TP, FP, _, _ = NewGetTPRPFromOnePatternset(patternset, val_case_patID_list[indexToFold],
                                                               val_case_AlarmSet_list[indexToFold],
                                                               val_case_PatientIDSet_list[indexToFold],
                                                               val_control_patID_list[indexToFold],
                                                               val_control_AlarmSet_list[indexToFold],
                                                               val_control_PatientIDSet_list[indexToFold])
                else:
                    TP = 0
                    FP = 0
                TP_temp = TP_temp + TP
                FP_temp = FP_temp + FP
            # print(TP_temp, FP_temp)
            TP_list.append(TP_temp / fold_num)
            FP_list.append(FP_temp / fold_num)
        print(TP_list)
        print(FP_list)

        plot_fp = FP_list.copy()
        plot_fp.sort()
        plot_TPFP = {}
        for fp_idx in range(len(FP_list)):
            plot_TPFP[FP_list[fp_idx]] = TP_list[fp_idx]
        plot_tp = []
        for sort_fp in plot_fp:
            plot_tp.append(plot_TPFP[sort_fp])
        # Plot_ROC(plot_fp, plot_tp)
        MinSup_TP.append(TP_list)
        MinSup_FP.append(FP_list)
        MinSup_FP_thresh.append(FP_thresh)

    folder_path = generate_path + 'minsup'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file = open(folder_path + '/MinSup_TP.txt', 'w')
    file.write(str(MinSup_TP))
    file.close()
    file = open(folder_path + '/MinSup_FP.txt', 'w')
    file.write(str(MinSup_FP))
    file.close()
    file = open(folder_path + '/MinSup_FP_thresh.txt', 'w')
    file.write(str(MinSup_FP_thresh))
    file.close()

    # each MINSUP has a ROC, then FPR_MAX_list = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    # select the maximum TPR under each FPR_MAX
    optimal_minsup_list, max_TTP_list, max_FFP_list, final_FP_thresh_list = GetOptimalMinSup(MinSup_list, FPR_MAX_list,
                                                                                             MinSup_TP,
                                                                                             MinSup_FP,
                                                                                             MinSup_FP_thresh)

    print(optimal_minsup_list)
    print(max_TTP_list)
    print(max_FFP_list)
    print(final_FP_thresh_list)
    optimal_folder_path = generate_path + 'minsup/optimal'
    if not os.path.exists(optimal_folder_path):
        os.makedirs(optimal_folder_path)
    file = open(optimal_folder_path + '/optimal_minsup_list.txt', 'w')
    file.write(str(optimal_minsup_list))
    file.close()
    file = open(optimal_folder_path + '/max_TTP_list.txt', 'w')
    file.write(str(max_TTP_list))
    file.close()
    file = open(optimal_folder_path + '/max_FFP_list.txt', 'w')
    file.write(str(max_FFP_list))
    file.close()
    file = open(optimal_folder_path + '/final_FP_thresh_list.txt',
                'w')
    file.write(str(final_FP_thresh_list))
    file.close()
    return optimal_minsup_list, final_FP_thresh_list


def GetTPRPFromOnePatternset(szOutput4Mafia, patternset, val_case_binaryCM, val_control_binaryCM):
    val_case_patID = [val + 1 for val in list(range(len(val_case_binaryCM)))]
    val_case_AlarmSet, val_case_PatientIDSet = parallel_zzzGetMFIfromMAFIA(szOutput4Mafia, val_case_binaryCM,
                                                                           val_case_patID)
    val_control_patID = [val + 1 for val in list(range(len(val_control_binaryCM)))]
    val_control_AlarmSet, val_control_PatientIDSet = parallel_zzzGetMFIfromMAFIA(szOutput4Mafia, val_control_binaryCM,
                                                                                 val_control_patID)
    case_pat_num = len(val_case_patID)
    control_pat_num = len(val_control_patID)
    TP_count = []
    FP_count = []
    pattern_list_tohighlight = []
    TP4EachPattern = []
    FP4EachPattern = []
    for pattern in patternset:
        if len(pattern) in val_case_AlarmSet.keys():
            if pattern in val_case_AlarmSet[len(pattern)]:
                index = val_case_AlarmSet[len(pattern)].index(pattern)
                TP_count.extend(val_case_PatientIDSet[len(pattern)][index])
                pattern_list_tohighlight.append(pattern)
                TP4EachPattern.append(len(val_case_PatientIDSet[len(pattern)][index]))
        if len(pattern) in val_control_AlarmSet.keys():
            if pattern in val_control_AlarmSet[len(pattern)]:
                index = val_control_AlarmSet[len(pattern)].index(pattern)
                FP_count.extend(val_control_PatientIDSet[len(pattern)][index])
                FP4EachPattern.append(len(val_control_PatientIDSet[len(pattern)][index]))
    # print(TP_count)
    TP = len(list(set(TP_count))) / case_pat_num
    FP = len(list(set(FP_count))) / control_pat_num
    return TP, FP, pattern_list_tohighlight, TP4EachPattern, FP4EachPattern


def GenerateOfflineResults(file_path, optimal_minsup_list, operating_point, FPR_MAX_list):
    # 最后用optimal的minsup在整个train上
    train_case_codematrix = np.loadtxt(
        file_path + "/binaryCM/train_case_codematrix.txt",
        delimiter=',')
    train_control_codematrix = np.loadtxt(
        file_path + "/binaryCM/train_control_codematrix.txt",
        delimiter=',')
    test_case_codematrix = np.loadtxt(
        file_path + "/binaryCM/test_case_codematrix.txt",
        delimiter=',')
    test_control_codematrix = np.loadtxt(
        file_path + "/binaryCM/test_control_codematrix.txt",
        delimiter=',')
    whole_train_case_locCM = transfer_BinaryCMTolocCM(train_case_codematrix)
    whole_train_case_locCM_output = open(
        file_path + "/locCM/whole_ards_train_locCM.txt", 'w',
        encoding='gbk')
    for row in whole_train_case_locCM:
        link = ' '
        row = [str(int(x)) for x in row]
        rowtxt = link.join(row)
        whole_train_case_locCM_output.write(rowtxt)
        whole_train_case_locCM_output.write('\n')
    for zero_row in range(train_case_codematrix.shape[0] - len(whole_train_case_locCM)):
        whole_train_case_locCM_output.write(str(100000 + zero_row))
        whole_train_case_locCM_output.write('\n')
    whole_train_case_locCM_output.close()

    superalarm_list = []
    for FPR_MAX_idx in range(len(FPR_MAX_list)):
        FPR_MAX = FPR_MAX_list[FPR_MAX_idx]
        print("under FPR max:", FPR_MAX)
        optimal_minsup = optimal_minsup_list[FPR_MAX_idx]
        # generate patterns set from train_case_codematrix
        finalOutput4Mafia = file_path + "/mafia_output/Output_final_" + str(
            optimal_minsup) + ".txt"
        finalMafiaCPP = mafia_path + ' -mfi ' + str(
            optimal_minsup) + ' -ascii ' + file_path + 'tokens/matrix/train_case_locCM.txt' + ' ' + finalOutput4Mafia
        print(finalMafiaCPP)
        run_final = os.system(finalMafiaCPP)
        # print(run_final)
        # get patterns set from train_case_codematrix
        # whole_train_case_patID = [val + 1 for val in list(range(len(train_case_codematrix)))]
        # whole_train_case_AlarmSet, whole_train_case_PatientIDSet = parallel_zzzGetMFIfromMAFIA(finalOutput4Mafia,
        #                                                                                       train_case_codematrix,
        #                                                                                       whole_train_case_patID)
        # print(whole_train_case_AlarmSet)
        # print(whole_train_case_PatientIDSet)
        # get patterns set from train_control_codematrix
        whole_train_control_patID = [val + 1 for val in list(range(len(train_control_codematrix)))]
        whole_train_control_AlarmSet, whole_train_control_PatientIDSet = parallel_zzzGetMFIfromMAFIA(finalOutput4Mafia,
                                                                                                     train_control_codematrix,
                                                                                                     whole_train_control_patID)
        # print(whole_train_control_AlarmSet)
        # print(whole_train_control_PatientIDSet)
        # whole_case_test_patID = [val + 1 for val in list(range(len(test_case_codematrix)))]
        # whole_case_test_AlarmSet, whole_case_test_PatientIDSet = parallel_zzzGetMFIfromMAFIA(finalOutput4Mafia,
        #                                                                                     test_case_codematrix,
        #                                                                                     whole_case_test_patID)

        # whole_control_test_patID = [val + 1 for val in list(range(len(test_control_codematrix)))]
        # whole_control_test_AlarmSet, whole_control_test_PatientIDSet = parallel_zzzGetMFIfromMAFIA(finalOutput4Mafia,
        #                                                                                           test_control_codematrix,
        #                                                                                           whole_control_test_patID)

        # 这里先在non上算fp
        FPSET, FPlist, Patternlist = GetPatternFP(whole_train_control_AlarmSet, whole_train_control_PatientIDSet,
                                                  whole_train_control_patID)
        # 然后threshold得到最后superalarm set
        superalarmset = GetPatternSetByFP(FPlist, Patternlist, operating_point[FPR_MAX_idx])
        print(len(superalarmset))
        # 最后算一下在test上的TP和FP
        test_TP, test_FP, pattern_list_tohighlight, TP4EachPattern = GetTPRPFromOnePatternset(
            finalOutput4Mafia, superalarmset, test_case_codematrix, test_control_codematrix)
        print("TP:", test_TP)
        print("FP:", test_FP)
        file = open(file_path + '/superalarm/superalarm_' + str(FPR_MAX) + '.txt', 'w')
        file.write(str(superalarmset))
        file.close()
        file = open(
            file_path + '/superalarm/pattern_list_tohighlight_' + str(FPR_MAX) + '.txt',
            'w')
        file.write(str(pattern_list_tohighlight))
        file.close()
        file = open(
            file_path + '/superalarm/TP4EachPattern_' + str(FPR_MAX) + '.txt',
            'w')
        file.write(str(TP4EachPattern))
        file.close()


def GenerateOfflineResults_alreadygotoutputversion(
        generate_path, optimal_minsup_list, operating_point, FPR_MAX_list):

    # finally, use optimal minsup to get superalarm set from the whole training set
    train_case_codematrix = np.loadtxt(
        generate_path + "tokens/matrix/train_case_codematrix.txt",
        delimiter=',')
    train_control_codematrix = np.loadtxt(
        generate_path + "tokens/matrix/train_control_codematrix.txt",
        delimiter=',')
    test_case_codematrix = np.loadtxt(
        generate_path + "tokens/matrix/test_case_codematrix.txt",
        delimiter=',')
    test_control_codematrix = np.loadtxt(
        generate_path + "tokens/matrix/test_control_codematrix.txt",
        delimiter=',')
    whole_train_case_locCM = transfer_BinaryCMTolocCM(train_case_codematrix)
    whole_train_case_locCM_output = open(
        generate_path + "tokens/matrix/train_case_locCM.txt", 'w',
        encoding='gbk')
    for row in whole_train_case_locCM:
        link = ' '
        row = [str(int(x)) for x in row]
        rowtxt = link.join(row)
        whole_train_case_locCM_output.write(rowtxt)
        whole_train_case_locCM_output.write('\n')
    for zero_row in range(train_case_codematrix.shape[0] - len(whole_train_case_locCM)):
        whole_train_case_locCM_output.write(str(100000 + zero_row))
        whole_train_case_locCM_output.write('\n')
    whole_train_case_locCM_output.close()

    folder_path = generate_path + '/superalarm'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for FPR_MAX_idx in range(len(FPR_MAX_list)):
        FPR_MAX = FPR_MAX_list[FPR_MAX_idx]
        print("under FPR max:", FPR_MAX)
        optimal_minsup = optimal_minsup_list[FPR_MAX_idx]
        # generate patterns set from train_case_codematrix
        finalOutput4Mafia = generate_path + "mafia_output/Output_final_" + str(
            optimal_minsup) + ".txt"
        finalMafiaCPP = mafia_path + ' -mfi ' + str(
            optimal_minsup) + ' -ascii ' + generate_path + 'tokens/matrix/train_case_locCM.txt' + ' ' + finalOutput4Mafia
        print(finalMafiaCPP)
        os.system(finalMafiaCPP)

        # get patterns set from train_control_codematrix
        whole_train_control_patID = [val + 1 for val in list(range(len(train_control_codematrix)))]
        whole_train_control_AlarmSet, whole_train_control_PatientIDSet = \
            parallel_zzzGetMFIfromMAFIA(finalOutput4Mafia, train_control_codematrix, whole_train_control_patID)

        # first calculate FP on train control patients
        FPSET, FPlist, Patternlist = GetPatternFP(whole_train_control_AlarmSet, whole_train_control_PatientIDSet,
                                                  whole_train_control_patID)
        # then get the final superalarm sets using the optimal minimum support
        superalarmset = GetPatternSetByFP(FPlist, Patternlist, operating_point[FPR_MAX_idx])
        print(len(superalarmset))
        # finally, calculate the TP and FP on test case patients
        test_TP, test_FP, pattern_list_tohighlight, TP4EachPattern = GetTPRPFromOnePatternset(
            finalOutput4Mafia, superalarmset, test_case_codematrix, test_control_codematrix)
        print("TP:", test_TP, "FP:", test_FP)

        file = open(folder_path + '/superalarm_' + str(FPR_MAX) + '.txt', 'w')
        file.write(str(superalarmset))
        file.close()
        file = open(
            folder_path + '/pattern_list_tohighlight_' + str(FPR_MAX) + '.txt',
            'w')
        file.write(str(pattern_list_tohighlight))
        file.close()
        file = open(
            folder_path + '/TP4EachPattern_' + str(FPR_MAX) + '.txt',
            'w')
        file.write(str(TP4EachPattern))
        file.close()


def GenerateOfflineResults_alreadyhavepatternsetversion(generate_path, FPR_MAX_list, optimal_minsup_list):
    mafia_output_path = generate_path + 'mafia_output/'
    # finally, use optimal minsup to get superalarm set from the whole training set
    test_case_codematrix = np.loadtxt(
        generate_path + "tokens/matrix/test_case_codematrix.txt",
        delimiter=',')
    test_control_codematrix = np.loadtxt(
        generate_path + "tokens/matrix/test_control_codematrix.txt",
        delimiter=',')

    folder_path = generate_path + '/superalarm'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    superalarm_path = folder_path + '/'
    for FPR_MAX_idx in range(len(FPR_MAX_list)):
        FPR_MAX = FPR_MAX_list[FPR_MAX_idx]
        print("under FPR max:", FPR_MAX)
        optimal_minsup = optimal_minsup_list[FPR_MAX_idx]
        # generate patterns set from train_case_codematrix
        finalOutput4Mafia = mafia_output_path + "Output_final_" + str(
            optimal_minsup) + ".txt"
        file_path = superalarm_path + 'superalarm_' + str(FPR_MAX) + '.txt'
        with open(file_path, "r") as file:
            contents = file.read()
        superalarmset = ast.literal_eval(contents)
        # finally, calculate the TP and FP on test case patients
        test_TP, test_FP, pattern_list_tohighlight, TP4EachPattern, FP4EachPattern = GetTPRPFromOnePatternset(
            finalOutput4Mafia, superalarmset, test_case_codematrix, test_control_codematrix)
        print("TP:", test_TP, "FP:", test_FP)

        file = open(
            folder_path + '/pattern_list_tohighlight_' + str(FPR_MAX) + '.txt',
            'w')
        file.write(str(pattern_list_tohighlight))
        file.close()
        file = open(
            folder_path + '/TP4EachPattern_' + str(FPR_MAX) + '.txt',
            'w')
        file.write(str(TP4EachPattern))
        file.close()
        file = open(
            folder_path + '/FP4EachPattern_' + str(FPR_MAX) + '.txt',
            'w')
        file.write(str(FP4EachPattern))
        file.close()
