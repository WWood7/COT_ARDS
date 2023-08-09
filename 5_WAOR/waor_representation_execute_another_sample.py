from waor_representation_functions_another_sample import *
from sklearn import metrics
import pandas as pd
import time
import multiprocessing as mp
from functools import partial

generate_path = '/Users/winnwu/projects/Hu_Lab/COT_project/generate/'
data_path = '/Users/winnwu/projects/Hu_Lab/COT_project/data/'
tokenarray_path = generate_path + 'tokenarray/'

# #
# # get WAOR for training set
# NumTrigCon = 10
# FPR_max = 0.15  # [0.02, 0.05, 0.1, 0.15, 0.2, 0.25,0.3,0.35,0.4]
# # get the ratio of case and control
# CaseOverSamRatio = get_ratio(generate_path, FPR_max)
# print(CaseOverSamRatio)
#
# # get the combinations of the parameters
# centerTimeinMins_list = [10, 30, 50, 60, 90]
# optFea_WAOR_list = {'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                     'b': [6, 3, 2, 1.5, 1.2, 1, 6. / 7, 0.75, 6. / 9, 0.6]}  # {'gama': 0.5, 'alpha': 1./2, 'beta':0.1}#
# combination = []
# for centerTimeinMins in centerTimeinMins_list:
#     for alpha in optFea_WAOR_list['a']:
#         for beta in optFea_WAOR_list['b']:
#             combination.append([centerTimeinMins, alpha, beta])
# print(len(combination))
#
# trainortest = 'train'
# for iscontrol in [0, 1]:
#     if iscontrol == 0:
#         caseorcontrol = 'case'
#     else:
#         caseorcontrol = 'control'
#     hitstruct = np.load(
#         tokenarray_path + caseorcontrol + '_' + trainortest + '_HitArray_dict_' + str(FPR_max)
#         + '_sparse.npy', allow_pickle=True).item()
#     patientid = list(hitstruct.keys())
#     toolbox_input = np.load(
#         tokenarray_path + caseorcontrol + '_' + trainortest + '_toolbox_input_' + str(FPR_max)
#         + '_sparse.npy', allow_pickle=True)
#     toolbox_input_df = pd.DataFrame({'PatID': toolbox_input[:, 0],
#                                                 'TimeToEnd': toolbox_input[:, 1],
#                                                 'HitOrNot': toolbox_input[:, 2]})
#     print('Generating ' + trainortest + caseorcontrol + ' GWAOR ...')
#     WAOR_pool(patient_list=patientid, trainortest=trainortest, caseorcontrol=caseorcontrol, iscontrol=iscontrol,
#                           hitstruct=hitstruct, toolbox_input_df=toolbox_input_df, NumTrigCon=NumTrigCon,
#                           CaseOverSamRatio=CaseOverSamRatio,
#                             combination=combination, FPR_max=FPR_max, file_path=generate_path)

# #
# # get WAOR for testing set
# NumTrigCon = 10
# FPR_max = 0.15  # [0.02, 0.05, 0.1, 0.15, 0.2, 0.25,0.3,0.35,0.4]
# CaseOverSamRatio = 0
# # get the combinations of the parameters
# centerTimeinMins_list = [10, 30, 50, 60, 90]
# optFea_WAOR_list = {'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                     'b': [6, 3, 2, 1.5, 1.2, 1, 6. / 7, 0.75, 6. / 9, 0.6]}  # {'gama': 0.5, 'alpha': 1./2, 'beta':0.1}#
# combination = []
# for centerTimeinMins in centerTimeinMins_list:
#     for alpha in optFea_WAOR_list['a']:
#         for beta in optFea_WAOR_list['b']:
#             combination.append([centerTimeinMins, alpha, beta])
# print(len(combination))
#
# trainortest = 'test'
# for iscontrol in [0, 1]:
#     if iscontrol == 0:
#         caseorcontrol = 'case'
#     else:
#         caseorcontrol = 'control'
#     hitstruct = np.load(
#         tokenarray_path + caseorcontrol + '_' + trainortest + '_HitArray_dict_' + str(FPR_max)
#         + '_sparse.npy', allow_pickle=True).item()
#     patientid = list(hitstruct.keys())
#     toolbox_input = np.load(
#         tokenarray_path + caseorcontrol + '_' + trainortest + '_toolbox_input_' + str(FPR_max)
#         + '_sparse.npy', allow_pickle=True)
#     toolbox_input_df = pd.DataFrame({'PatID': toolbox_input[:, 0],
#                                                 'TimeToEnd': toolbox_input[:, 1],
#                                                 'HitOrNot': toolbox_input[:, 2]})
#     print('Generating ' + trainortest + caseorcontrol + ' GWAOR ...')
#     WAOR_pool(patient_list=patientid, trainortest=trainortest, caseorcontrol=caseorcontrol, iscontrol=iscontrol,
#                           hitstruct=hitstruct, toolbox_input_df=toolbox_input_df, NumTrigCon=NumTrigCon,
#                           CaseOverSamRatio=CaseOverSamRatio,
#                             combination=combination, FPR_max=FPR_max, file_path=generate_path)


# find the optimal hyperparameters for WAOR using CV

centerTimeinMins_list = [10, 30, 50, 60, 90]
optFea_WAOR_list = {'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'b': [6, 3, 2, 1.5, 1.2, 1, 6. / 7, 0.75, 6. / 9, 0.6]}

combination = []
combination_index_list = list(range(10))
for centerTimeinMins in centerTimeinMins_list:
    for alpha in optFea_WAOR_list['a']:
        for beta in optFea_WAOR_list['b']:
            combination.append([centerTimeinMins, alpha, beta])
get_params_cv(combination_index_list, 0.15, combination, generate_path)


# # get the WAOR representation on the test set
# if __name__ == "__main__":
#     file_path = '/Users/winnwu/projects/emory-hu lab/workshop/generate/'
#     centerTimeinMins_list = [10]
#     alpha = [4]
#     beta = [1]
#     plambda = [0.1]
#     FPR_max_list = [0.05]  # [0.02, 0.05, 0.1, 0.15, 0.2, 0.25,0.3,0.35,0.4]
#     FPR_max = 0.05
#     optFea_WAOR_list = {'a': [4], 'b': [1]}
#     trainortest = 'test'
#     iscontrol = 1
#     if iscontrol == 0:
#         caseorcontrol = 'case'
#     else:
#         caseorcontrol = 'control'
#
#     combination = []
#     for centerTimeinMins in centerTimeinMins_list:
#         for alpha in optFea_WAOR_list['a']:
#             for beta in optFea_WAOR_list['b']:
#                 combination.append([centerTimeinMins, alpha, beta])
#     print(len(combination))
#     hitstruct = np.load(
#         file_path + 'tokenarray/' + caseorcontrol + '_' + trainortest + '_HitArray_dict_' + str(
#             FPR_max) + '_sparse.npy',
#         allow_pickle=True).item()
#     patientid = list(hitstruct.keys())
#     toolbox_input = np.load(
#         file_path + 'tokenarray/' + caseorcontrol + '_' + trainortest + '_toolbox_input_' + str(
#             FPR_max) + '_sparse.npy',
#         allow_pickle=True)
#     case_train_toolbox_input_df = pd.DataFrame({'PatID': toolbox_input[:, 0],
#                                                 'TimeToEnd': toolbox_input[:, 1],
#                                                 'HitOrNot': toolbox_input[:, 2]})
#     print(len(patientid))
#     print('Generating ' + trainortest + caseorcontrol + ' GWAOR ...')
#
#     tic = time.time()
#     partial_work = partial(WAOR_pool, trainortest=trainortest, caseorcontrol=caseorcontrol, iscontrol=iscontrol,
#                            case_train_hitstruct=hitstruct,
#                            case_train_toolbox_input_df=case_train_toolbox_input_df, NumTrigCon=5,
#                            CaseOverSamRatio=0.5,
#                            combination=combination, FPR_max=FPR_max, file_path=file_path)
#     processnum = 8
#     pool = mp.Pool(processnum)
#     patstep = round(len(patientid) / processnum)
#     patis_list = [patientid[i * patstep:i * patstep + patstep] for i in range(processnum - 1)]
#     patis_list.append(patientid[(processnum - 1) * patstep:])
#     pool.map(partial_work, patis_list)
#     pool.close()
#     pool.join()
#
# # get the final classification results on the test set
# file_path = '/Users/winnwu/projects/emory-hu lab/workshop/generate/'
# centerTimeinMins_list = [10]
# alpha = [4]
# beta = [1]
# plambda = [0.1]
# FPR_max_list = [0.05]  # [0.02, 0.05, 0.1, 0.15, 0.2, 0.25,0.3,0.35,0.4]
# FPR_max = 0.05
# optFea_WAOR_list = {'a': [4], 'b': [1]}
# # first train the classifier on the training WAORs
# # get the case patients' data
# combination_idex = 142
# case_train_data_list = glob.glob(file_path + 'WAOR_files/case_trainData_allFea_GWAOR_' + str(FPR_max) +
#                                  '_sparse_comb_' + str(combination_idex) + 'patstart*.npy')
# case_train_list = []
# for file in case_train_data_list:
#     data = np.load(file, allow_pickle=True).item()
#     case_train_list.append(data)
# CA_trainData_allFea_GWAOR = vstack(case_train_list)
#
# # get the control patients' data
# control_train_data_list = glob.glob(file_path + 'WAOR_files/control_trainData_allFea_GWAOR_' + str(FPR_max) +
#                                     '_sparse_comb_' + str(combination_idex) + 'patstart*.npy')
# control_train_list = []
# for file in control_train_data_list:
#     data = np.load(file, allow_pickle=True).item()
#     control_train_list.append(data)
# Control_trainData_allFea_GWAOR = vstack(control_train_list)
# CA_trainData_allFea_GWAOR_dense = CA_trainData_allFea_GWAOR.todense()
# Control_trainData_allFea_GWAOR_dense = Control_trainData_allFea_GWAOR.todense()
# alltrain = np.vstack((CA_trainData_allFea_GWAOR_dense, Control_trainData_allFea_GWAOR_dense))
# X_train = alltrain[:, :-2]
# y_train = np.ravel(alltrain[:, -2])
# max_xtrain = np.max(X_train, axis=0)
# max_xtrain[max_xtrain == 0] = 1
# X_train_norm = (X_train - np.min(X_train, axis=0)) / (max_xtrain - np.min(X_train, axis=0))
# X_train_norm = np.float16(X_train_norm)
# model = LogisticRegression(solver='liblinear', max_iter=100, penalty='l1', tol=1e-4,
#                            C=0.1).fit(np.asarray(X_train), y_train)
#
# # then test the classifier on the test WAORs
# # get the case patients' data
# case_test_data_list = glob.glob(file_path + 'WAOR_files/case_testData_allFea_GWAOR_' + str(FPR_max) +
#                                 '_sparse_comb_' + str(0) + 'patstart*.npy')
# case_test_list = []
# for file in case_test_data_list:
#     data = np.load(file, allow_pickle=True).item()
#     case_test_list.append(data)
# CA_testData_allFea_GWAOR = vstack(case_test_list)
#
# # get the control patients' data
# control_test_data_list = glob.glob(file_path + 'WAOR_files/control_testData_allFea_GWAOR_' + str(FPR_max) +
#                                    '_sparse_comb_' + str(0) + 'patstart*.npy')
# control_test_list = []
# for file in control_test_data_list:
#     data = np.load(file, allow_pickle=True).item()
#     control_test_list.append(data)
# Control_testData_allFea_GWAOR = vstack(control_test_list)
#
# CA_testData_allFea_GWAOR_dense = CA_testData_allFea_GWAOR.todense()
# Control_testData_allFea_GWAOR_dense = Control_testData_allFea_GWAOR.todense()
# alltest = np.vstack((CA_testData_allFea_GWAOR_dense, Control_testData_allFea_GWAOR_dense))
# X_test = alltest[:, :-2]
# y_test = np.ravel(alltest[:, -2])
# max_xtest = np.max(X_test, axis=0)
# max_xtest[max_xtest == 0] = 1
# X_test_norm = (X_test - np.min(X_test, axis=0)) / (max_xtest - np.min(X_test, axis=0))
# X_test_norm = np.float16(X_test_norm)
# y_pred = model.predict(np.asarray(X_test))
# print(y_pred)
# y_pred_prob = model.predict_proba(np.asarray(X_test))
#
# y_pred_prob = y_pred_prob[:, 1]
# print(y_pred_prob)
# y_pred_label = y_pred_prob > 0.61
# print(y_pred_label)
# print('test accuracy: ', metrics.accuracy_score(y_test, y_pred))
# true_pos = 0
# case_list = [element[0][0, 0] for element in CA_testData_allFea_GWAOR_dense[:, -1]]
# for i in np.unique(case_list):
#     # get the index of the patients in the test set
#     index = np.where(case_list == i)
#     if sum(y_pred_label[index]) > 0:
#         true_pos = true_pos + 1
# print('true positive: ', true_pos)
# print('false negative:', len(np.unique(case_list)) - true_pos)
# control_list = [element[0][0, 0] for element in Control_testData_allFea_GWAOR_dense[:, -1]]
# false_pos = 0
# for i in np.unique(control_list):
#     # get the index of the patients in the test set
#     index = np.where(control_list == i)
#     if sum(y_pred_label[index]) > 0:
#         false_pos = false_pos + 1
# print('false positive: ', false_pos)
# print('true negative:', len(np.unique(control_list)) - false_pos)


