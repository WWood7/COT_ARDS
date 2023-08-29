import pandas as pd
import os
import numpy as np
import ast
import time
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial

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


def GenerateCodeMatrix4Whole(generate_path):
    tokenid_path = generate_path.replace('mimiciv/', '')
    # get segments
    test_case_seg = pd.read_csv(generate_path + 'segments/case_segs.csv')
    test_case_seg['PatientID'] = test_case_seg['icustay_id']
    test_case = test_case_seg[['PatientID']]
    test_control_seg = pd.read_csv(generate_path + 'segments/control_segs.csv')
    test_control_seg['PatientID'] = test_control_seg['seg_id']
    test_control = test_control_seg[['PatientID']]

    # get all the token inputs
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

    tokenid_num = pd.read_csv(tokenid_path + 'tokens/matrix/tokenid_loc.csv')
    folder_path = generate_path + 'tokens/matrix'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
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
    # print(TP_count)
    TP = len(list(set(TP_count))) / case_pat_num
    FP = len(list(set(FP_count))) / control_pat_num
    return TP, FP, pattern_list_tohighlight, TP4EachPattern


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
        tic_temp = time.time()
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
        # print(time.time()-tic_temp,'one lenth')
    # print(time.time() - tic22, 'last half of zzz')
    return AlarmSetMFIFromMafia, patientIDSet


def GenerateOfflineResults(generate_path, FPR_MAX_list, optimal_minsup_list):
    mafia_output_path = generate_path.replace('mimiciv/', '') + 'mafia_output/'
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
    superalarm_path = '/Users/winnwu/projects/Hu_lab/COT_project/generate/superalarm/'
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
        test_TP, test_FP, pattern_list_tohighlight, TP4EachPattern = GetTPRPFromOnePatternset(
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
