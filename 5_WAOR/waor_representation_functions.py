import numpy as np
from scipy.sparse import csr_matrix, vstack, save_npz, load_npz
from scipy.stats import gamma
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, GroupKFold
import os
import random

def get_ratio(generate_path, FPR_max):
    case = np.load(generate_path + 'tokenarray/case_train_toolbox_input_' + str(FPR_max) +
                   '_sparse.npy', allow_pickle=True)
    control = np.load(generate_path + 'tokenarray/control_train_toolbox_input_' + str(FPR_max) +
                   '_sparse.npy', allow_pickle=True)
    enc_case = 0
    for i in np.unique(case[:, 0]):
        trigger_num = sum(case[np.where(case[:, 0] == i)][:, 2])
        if trigger_num > 0:
            enc_case += 1

    enc_control = 0
    for i in np.unique(control[:, 0]):
        trigger_num = sum(control[np.where(control[:, 0] == i)][:, 2])
        if trigger_num > 0:
            enc_control += 1
    return enc_control / enc_case


def sampleTriggers(onecase_hitstruct_hitarray, IsUniformDraw, NumTrigCon, CaseOverSamRatio,
                   centerTimeinMins, HitTIMEtoEND, trainortest):
    if trainortest == 'train':
        # IsUniformDraw indicates if the patient is a case patient or a control patient
        # 0 - case, 1- control
        if IsUniformDraw == 1:
            numofDraws = NumTrigCon
        else:
            numofDraws = max(round(NumTrigCon * CaseOverSamRatio), 1)

        # RX: add uniform Draw option for case as well
        if centerTimeinMins == np.inf:
            IsUniformDraw = 1

        # find indices of triggers, in HitArray each alarm generate a vector of
        # 1 or 0, only those with at least 1 in the vector counts as a hit
        idxofTriggers = np.nonzero(np.ones((1, onecase_hitstruct_hitarray.shape[0])) @ onecase_hitstruct_hitarray)[1]
        numofTriggers = len(idxofTriggers)
        print('numofTriggers: ', numofTriggers)

        # nothing to sample from
        if numofTriggers == 0:
            idxofSamples = []
            return idxofSamples

        if IsUniformDraw == 1:
            idx = np.random.permutation(numofTriggers)
            if numofTriggers <= numofDraws:
                idxofSamples = idxofTriggers
            else:
                # draw numofDraws samples from idxofTriggers randomly
                idxofSamples = idxofTriggers[idx[:numofDraws]]
        else:
            # what on earth is this if part doing?
            if np.isnan(centerTimeinMins):
                idx = numofTriggers - np.fix(np.random.exponential(numofDraws, (numofDraws, 1)))
                idxofSamples = idxofTriggers[max(1, idx)]
            else:
                # Obtain the desired subset of HitTIMEtoEND.
                t = HitTIMEtoEND[idxofTriggers]

                # Generate exponential random variables.
                t1 = np.random.exponential(centerTimeinMins / 60, numofDraws)

                # Compute absolute differences between each sample in t1 and all elements in t.
                temp_t = np.abs(t - t1[:, None])

                # Find the index in t of the closest element to each sample in t1.
                tidx = np.argmin(temp_t, axis=1)

                # Get elements from idxofTriggers at the indices specified by tidx.
                idxofSamples = idxofTriggers[tidx]

    else:
        idxofTriggers = np.nonzero(np.ones((1, onecase_hitstruct_hitarray.shape[0])) @ onecase_hitstruct_hitarray)[1]
        numofTriggers = len(idxofTriggers)

        # nothing to sample from
        if numofTriggers == 0:
            idxofSamples = []
        else:
            idxofSamples = idxofTriggers
    print('idxofSamples: ', len(idxofSamples))
    return idxofSamples


def weightingFuncGWAOR(hitArray, deltaT, a, b):
    return hitArray @ gamma.pdf(deltaT, a=a, scale=1 / b)


def WAOR_fuc(centerTimeinMins, alpha, beta, iscontrol, onecase_hitstruct_hitarray, onecase_hitstruct_hitt, patid,
             toolbox_input_df, NumTrigCon, CaseOverSamRatio, trainortest):

    idxofTriggerSamples = sampleTriggers(onecase_hitstruct_hitarray, iscontrol, NumTrigCon, CaseOverSamRatio,
                                         centerTimeinMins,
                                         toolbox_input_df['TimeToEnd'].loc[
                                             toolbox_input_df['PatID'] == patid].values, trainortest)
    temp_WAOR = {}
    if len(idxofTriggerSamples) != 0:
        fea_GWAOR = np.zeros((len(idxofTriggerSamples), onecase_hitstruct_hitarray.shape[0]))
        for i in range(len(idxofTriggerSamples)):
            k = idxofTriggerSamples[i]
            fea_GWAOR[i] = weightingFuncGWAOR(onecase_hitstruct_hitarray[:, :k+1],
                                              onecase_hitstruct_hitt[k] - onecase_hitstruct_hitt[
                                                                          :k+1], alpha, beta)
        if iscontrol == 1:
            temp_WAOR = np.hstack((fea_GWAOR, np.zeros((len(idxofTriggerSamples), 1)),
                                   np.ones((len(idxofTriggerSamples), 1)) * patid))
        else:
            temp_WAOR = np.hstack((fea_GWAOR, np.ones((len(idxofTriggerSamples), 1)),
                                   np.ones((len(idxofTriggerSamples), 1)) * patid))

    return temp_WAOR


def WAOR_pool(patient_list, trainortest, caseorcontrol, hitstruct, toolbox_input_df, NumTrigCon,
              CaseOverSamRatio, combination, FPR_max, file_path, iscontrol):
    # combination: combinations of parameters [0]cneterTimeinMins [1]alpha [2]beta

    for comb_idx in range(len(combination)):
        print(comb_idx)
        case_trainData_allFea_GWAOR = {}
        comb = combination[comb_idx]
        centerTimeinMins = comb[0]
        alpha = comb[1]
        beta = comb[2]

        # patient_list: list of patients' ids
        for j in patient_list:
            onecase_hitstruct_hitarray = hitstruct[j]['sparseHitArray'].todense()
            onecase_hitstruct_hitt = hitstruct[j]['HitT']

            WAOR4OneComb = WAOR_fuc(centerTimeinMins, alpha, beta, iscontrol, onecase_hitstruct_hitarray,
                                    onecase_hitstruct_hitt, j,
                                    toolbox_input_df, NumTrigCon, CaseOverSamRatio, trainortest)
            if len(WAOR4OneComb) != 0:
                if comb_idx in case_trainData_allFea_GWAOR.keys():
                    case_trainData_allFea_GWAOR[comb_idx].append(WAOR4OneComb)
                else:
                    case_trainData_allFea_GWAOR[comb_idx] = [WAOR4OneComb]

        if comb_idx in case_trainData_allFea_GWAOR.keys():
            case_TrainData_GWAOR = csr_matrix(np.concatenate(case_trainData_allFea_GWAOR[comb_idx]))
            if not os.path.exists(file_path + 'WAOR_files'):
                os.makedirs(file_path + 'WAOR_files')
            save_npz(
                file_path + 'WAOR_files/' + caseorcontrol + '_' + trainortest + 'Data_allFea_GWAOR_' + str(
                    FPR_max) + '_sparse_comb_' + str(comb_idx) + '.npz',
                case_TrainData_GWAOR)



