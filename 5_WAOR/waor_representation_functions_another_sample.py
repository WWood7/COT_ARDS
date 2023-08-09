import numpy as np
from scipy.sparse import csr_matrix, vstack, save_npz, load_npz
from scipy.stats import gamma
from sklearn.preprocessing import normalize
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
                print('t: ', t)

                # evaluate the exponential distribution at t
                pdf = np.exp(-t / (centerTimeinMins / 60)) / (centerTimeinMins / 60)
                if sum(pdf) == 0:
                    idxofSamples = []

                else:
                    weight = pdf / sum(pdf)
                    idxofSamples = np.random.choice(idxofTriggers, size=numofDraws, p=weight, replace=True)

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

    for comb_idx in range(1):
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
            if not os.path.exists(file_path + 'WAOR_files_another'):
                os.makedirs(file_path + 'WAOR_files_another')
            save_npz(
                file_path + 'WAOR_files_another/' + caseorcontrol + '_' + trainortest + 'Data_allFea_GWAOR_' + str(
                    FPR_max) + '_sparse_comb_' + str(comb_idx) + '.npz',
                case_TrainData_GWAOR)


def get_params_cv(combination_idex_list, FPR_max, combination, generate_path):
    auc_list = []
    acc_list = []
    for combination_idex in combination_idex_list:
        centerTimeinMins = combination[combination_idex][0]
        alpha = combination[combination_idex][1]
        beta = combination[combination_idex][2]
        print(combination[combination_idex])

        # load data
        train_case_data = (
            load_npz(generate_path + 'WAOR_files_another/case_trainData_allFea_GWAOR_' + str(FPR_max) +
                     '_sparse_comb_' + str(combination_idex) + '.npz').
            toarray())
        train_control_data = (
            load_npz(generate_path + 'WAOR_files_another/control_trainData_allFea_GWAOR_' + str(FPR_max) +
                     '_sparse_comb_' + str(combination_idex) + '.npz').
            toarray())
        train_data = np.concatenate((train_case_data, train_control_data))

        # normalize data
        train_X = train_data[:, :-2]
        normalized_train_X = normalize(train_X, axis=0, norm='max')
        train_y = train_data[:, -2]

        # define a spliter for cross validation
        # we don't want data from the same patient in both training and validation sets
        indices_1 = np.where(train_data[:, -2] == 1)[0]
        indices_0 = np.where(train_data[:, -2] == 0)[0]
        group_1 = train_data[train_data[:, -2] == 1][:, -1]
        group_0 = train_data[train_data[:, -2] == 0][:, -1]
        gkf = GroupKFold(n_splits=5)
        splits_1 = gkf.split(indices_1, groups=group_1)
        splits_0 = gkf.split(indices_0, groups=group_0)
        combined_splits = []
        for (train_idx_0, val_idx_0), (train_idx_1, val_idx_1) in zip(splits_0, splits_1):
            train_idx = np.concatenate([indices_0[train_idx_0], indices_1[train_idx_1]])
            val_idx = np.concatenate([indices_0[val_idx_0], indices_1[val_idx_1]])
            combined_splits.append((train_idx, val_idx))

        scoring = {'AUC': 'roc_auc', 'Accuracy': 'accuracy'}

        # load the data
        # get the scores
        clf = LogisticRegression()
        scores = cross_validate(clf, normalized_train_X, train_y, scoring=scoring, cv=combined_splits)
        auc_list.append(scores['test_AUC'].mean())
        acc_list.append(scores['test_Accuracy'].mean())
        print(scores['test_Accuracy'].mean())
        print(scores['test_AUC'].mean())

    # get the best parameters
    best_combination_idex_1 = combination_idex_list[np.argmax(auc_list)]
    best_combination_idex_2 = combination_idex_list[np.argmax(acc_list)]
    print('best_combination_auc: ', combination[best_combination_idex_1])
    print('best_combination_acc: ', combination[best_combination_idex_2])
    print('centerTimeinMins: ', combination[best_combination_idex_1][0])
    print('alpha: ', combination[best_combination_idex_1][1])
    print('beta: ', combination[best_combination_idex_1][2])



