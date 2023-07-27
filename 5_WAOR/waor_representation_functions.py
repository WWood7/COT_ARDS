import numpy as np
from scipy.sparse import csr_matrix, vstack, save_npz, load_npz
from scipy.stats import gamma
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import glob
import os

def sampleTriggers(onecase_hitstruct_hitarray, IsUniformDraw, NumTrigCon, CaseOverSamRatio,
                   centerTimeinMins, HitTIMEtoEND, trainortest):
    if trainortest == 'train':
        # IsUniformDraw indicates if the patient is a case patient or a control patient
        # 0 - case, 1- control
        if IsUniformDraw == 1:
            numofDraws = NumTrigCon
        else:
            numofDraws = round(NumTrigCon * CaseOverSamRatio)

        # RX: add uniform Draw option for case as well
        if centerTimeinMins == np.inf:
            IsUniformDraw = 1

        # find indices of triggers, in HitArray each alarm generate a vector of
        # 1 or 0, only those with at least 1 in the vector counts as a hit
        idxofTriggers = np.nonzero(np.ones((1, onecase_hitstruct_hitarray.shape[0])) @ onecase_hitstruct_hitarray)[1]
        numofTriggers = len(idxofTriggers)

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
                # improvement provided by GPT
                # Obtain the desired subset of HitTIMEtoEND.
                t = HitTIMEtoEND[idxofTriggers]

                # Generate exponential random variables.
                t1 = np.random.exponential(centerTimeinMins / 60, numofDraws)

                # Compute absolute differences between each sample in t1 and all elements in t.
                temp_t = np.abs(t - t1[:, None])

                # Find the index in t of the closest element to each sample in t1.
                tidx = np.argmin(temp_t, axis=1)

                # Get unique elements from idxofTriggers at the indices specified by tidx.
                idxofSamples = np.unique(idxofTriggers[tidx])

    else:
        idxofTriggers = np.nonzero(np.ones((1, onecase_hitstruct_hitarray.shape[0])) @ onecase_hitstruct_hitarray)[1]
        numofTriggers = len(idxofTriggers)

        # nothing to sample from
        if numofTriggers == 0:
            idxofSamples = []
        else:
            idxofSamples = idxofTriggers

    return idxofSamples


def weightingFuncGWAOR(hitArray, deltaT, a, b):
    return hitArray @ gamma.pdf(deltaT, a=a, scale=1 / b)


def WAOR_fuc(centerTimeinMins, alpha, beta, iscontrol, onecase_hitstruct_hitarray, onecase_hitstruct_hitt, patid,
             case_train_toolbox_input_df, NumTrigCon, CaseOverSamRatio, trainortest):

    idxofTriggerSamples = sampleTriggers(onecase_hitstruct_hitarray, iscontrol, NumTrigCon, CaseOverSamRatio,
                                         centerTimeinMins,
                                         case_train_toolbox_input_df['TimeToEnd'].loc[
                                             case_train_toolbox_input_df['PatID'] == patid].values, trainortest)
    temp_WAOR = {}
    if len(idxofTriggerSamples) != 0:
        fea_GWAOR = np.zeros((len(idxofTriggerSamples), onecase_hitstruct_hitarray.shape[0]))
        for i in range(len(idxofTriggerSamples)):
            k = idxofTriggerSamples[i]
            fea_GWAOR[i] = weightingFuncGWAOR(onecase_hitstruct_hitarray[:, :k],
                                              onecase_hitstruct_hitt[k] - onecase_hitstruct_hitt[
                                                                          :k], alpha, beta)
        if iscontrol == 1:
            temp_WAOR = np.hstack((fea_GWAOR, np.zeros((len(idxofTriggerSamples), 1)),
                                   np.ones((len(idxofTriggerSamples), 1)) * patid))
        else:
            temp_WAOR = np.hstack((fea_GWAOR, np.ones((len(idxofTriggerSamples), 1)),
                                   np.ones((len(idxofTriggerSamples), 1)) * patid))

    return temp_WAOR


def WAOR_pool(patient_list, trainortest, caseorcontrol, case_train_hitstruct, case_train_toolbox_input_df, NumTrigCon,
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
            onecase_hitstruct_hitarray = case_train_hitstruct[j]['sparseHitArray'].todense()
            onecase_hitstruct_hitt = case_train_hitstruct[j]['HitT']

            WAOR4OneComb = WAOR_fuc(centerTimeinMins, alpha, beta, iscontrol, onecase_hitstruct_hitarray,
                                    onecase_hitstruct_hitt, j,
                                    case_train_toolbox_input_df, NumTrigCon, CaseOverSamRatio, trainortest)
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
                    FPR_max) + '_sparse_comb_' + str(comb_idx) + 'patstart' + str(patient_list[0]) + 'To' + str(
                    patient_list[-1]) + '.npz',
                case_TrainData_GWAOR)


def resplit_data(casedata, controldata):
    #print(casedata)
    #print(controldata)
    splitRatio = 0.8
    #print(casedata[:,-1].shape)
    caPat = np.unique(casedata[:, -1].A).reshape((-1, 1))
    #print('when split', caPat)
    #print(len(caPat))
    conPat = np.unique(controldata[:, -1].A).reshape((-1, 1))
    caIdx = np.random.permutation(len(caPat))
    conIdx = np.random.permutation(len(conPat))
    #print(caPat.shape)
    #print(conPat.shape)

    caInd_train = caIdx[:round(splitRatio*len(caPat))]
    caInd_val = caIdx[round(splitRatio * len(caPat)):]
    conInd_train = conIdx[:round(splitRatio * len(conPat))]
    conInd_val = conIdx[round(splitRatio * len(conPat)):]
    #print(caInd_train.shape)
    #print(caInd_val.shape)

    caPat_train = caPat[caInd_train]
    caPat_val = caPat[caInd_val]
    conPat_train = conPat[conInd_train]
    conPat_val = conPat[conInd_val]
    #print(caPat_train.shape)
    #print(caPat_val.shape)

    cadata_train = casedata[[True if val in caPat_train else False for val in casedata[:,-1]], :]
    cadata_val = casedata[[True if val in caPat_val else False for val in casedata[:,-1]], :]
    condata_train = controldata[[True if val in conPat_train else False for val in controldata[:,-1]], :]
    condata_val = controldata[[True if val in conPat_val else False for val in controldata[:,-1]], :]
    #print(cadata_train.shape)
    #print(cadata_val.shape)

    allTrain = np.vstack((cadata_train, condata_train))
    allVal = np.vstack((cadata_val, condata_val))
    #print(allTrain.shape)

    final_train = allTrain[:, :-2]
    final_val = allVal[:, :-2]
    final_train_label = np.ravel(allTrain[:, -2])
    final_val_label = np.ravel(allVal[:, -2])
    #print(final_train.shape)
    #print(final_train_label.shape)

    return final_train, final_train_label, final_val, final_val_label


def Train_pool(combination_idex_list, FPR_max, combination, file_path):
    lambda_list = np.logspace(-1.5, -1, num=4)  # logspace(-3,-1,50);
    for combination_idex in combination_idex_list:
        centerTimeinMins = combination[combination_idex][0]
        fold = combination[combination_idex][1]
        alpha = combination[combination_idex][2]
        beta = combination[combination_idex][3]
        print(combination[combination_idex])

        # get the case patients' data
        case_train_data_list = glob.glob(file_path + 'WAOR_files/case_trainData_allFea_GWAOR_' + str(FPR_max) +
                                    '_sparse_comb_' + str(combination_idex) + 'patstart*.npy')
        case_train_list = []
        for file in case_train_data_list:
            data = np.load(file, allow_pickle=True).item()
            case_train_list.append(data)
        CA_trainData_allFea_GWAOR = vstack(case_train_list)

        # get the control patients' data
        control_train_data_list = glob.glob(file_path + 'WAOR_files/control_trainData_allFea_GWAOR_' + str(FPR_max) +
                                    '_sparse_comb_' + str(combination_idex) + 'patstart*.npy')
        control_train_list = []
        for file in control_train_data_list:
            data = np.load(file, allow_pickle=True).item()
            control_train_list.append(data)
        Control_trainData_allFea_GWAOR = vstack(control_train_list)

        #print(CA_trainData_allFea_GWAOR.shape)
        #print(Control_trainData_allFea_GWAOR.shape)
        CA_trainData_allFea_GWAOR_dense = CA_trainData_allFea_GWAOR.todense()
        Control_trainData_allFea_GWAOR_dense = Control_trainData_allFea_GWAOR.todense()
        trigger_ca_train = len(np.unique(CA_trainData_allFea_GWAOR_dense[:, -1].A))
        notrigger_ca_train = 1622 - trigger_ca_train
        #print(trigger_ca_train, notrigger_ca_train)
        notrigger_ca_train_waor = round(
            notrigger_ca_train * (CA_trainData_allFea_GWAOR_dense.shape[0] / trigger_ca_train) * 0.2)
        trigger_con_train = len(np.unique(Control_trainData_allFea_GWAOR_dense[:, -1].A))
        notrigger_con_train = 885 - trigger_con_train
        #print(trigger_con_train, notrigger_con_train)
        notrigger_con_train_waor = round(notrigger_con_train * (
                Control_trainData_allFea_GWAOR_dense.shape[0] / trigger_con_train) * 0.2)

        X_train, y_train, X_valid, y_valid = resplit_data(
            CA_trainData_allFea_GWAOR_dense,
            Control_trainData_allFea_GWAOR_dense)
        # print(X_train.shape,y_train.shape)

        max_xtrain = np.max(X_train, axis=0)
        max_xtrain[max_xtrain==0]=1
        X_train_norm = (X_train - np.min(X_train, axis=0)) / (max_xtrain - np.min(X_train, axis=0))
        X_train_norm = np.float16(X_train_norm)
        #print(X_train_norm.shape)
        max_xval = np.max(X_valid, axis=0)
        max_xval[max_xval == 0] = 1
        X_valid_norm = (X_valid - np.min(X_valid, axis=0)) / (max_xval - np.min(X_valid, axis=0))
        X_valid_norm = np.float16(X_valid_norm)
        temp_auc = []#np.zeros(len(lambda_list))
        #temp_score = np.zeros((len(y_valid), len(lambda_list)))
        #y_valid_temp = np.vstack((y_valid.reshape(-1, 1), np.zeros(notrigger_con_train_waor).reshape(-1, 1),
        #                          np.ones(notrigger_ca_train_waor).reshape(-1, 1)))
        for lamda_idx in range(len(lambda_list)):
            # tic = time.time()
            model = LogisticRegression(solver='liblinear', max_iter=100, penalty='l1', tol=1e-4,
                                       C=1 / lambda_list[lamda_idx]).fit(np.asarray(X_train_norm), y_train)
            score = model.predict_proba(np.asarray(X_valid_norm))
            #temp_score[:, lamda_idx] = score[:, 1]

            #score_temp = np.vstack((score[:, 1].reshape(-1, 1),
            #                        np.zeros(notrigger_con_train_waor + notrigger_ca_train_waor).reshape(-1, 1)))
            #print(score[:, 1].shape,'+',notrigger_con_train_waor,'+',notrigger_ca_train_waor,'=',y_valid.shape,'==',score_temp.shape)
            fpr, tpr, thresholds = metrics.roc_curve(y_valid, score[:, 1], pos_label=1)
            temp_auc.append(metrics.auc(fpr, tpr))
            # print('One LR:', time.time() - tic)
            # print('Hello')
        print(temp_auc)
        print("Saving...")
        np.save(file_path + 'WAOR_modelfiles/trainedModel_LR_maxFPR_' + str(FPR_max) + '_Performance_AUC_norm' + str(
            centerTimeinMins) + '_fold_' + str(fold) + '_alpha_'+str(alpha)+'_beta_'+str(beta)+'.npy', temp_auc)
        #np.save(file_path+generatedfile_path+'WAOR_modelfiles/trainedModel_LR_maxFPR_' + str(FPR_max) + '_Performance_Score_norm' + str(
        #    centerTimeinMins) + '_fold_' + str(fold) + '_alpha_'+str(alpha)+'_beta_'+str(beta)+'.npy', score[:,1])
        #np.save(file_path+generatedfile_path+'WAOR_modelfiles/trainedModel_LR_maxFPR_' + str(FPR_max) + '_Performance_True_norm' + str(
        #    centerTimeinMins) + '_fold_' + str(fold) + '_alpha_'+str(alpha)+'_beta_'+str(beta)+'.npy', y_valid)



