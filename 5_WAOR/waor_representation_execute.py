from waor_representation_functions import *
from sklearn import metrics
import pandas as pd
import time
import multiprocessing as mp
from functools import partial

generate_path = '/Users/winnwu/projects/Hu_Lab/COT_project/generate/'
data_path = '/Users/winnwu/projects/Hu_Lab/COT_project/data/'
tokenarray_path = generate_path + 'tokenarray/'

#
# get WAOR for training set
NumTrigCon = 10
FPR_max = 0.25  # [0.02, 0.05, 0.1, 0.15, 0.2, 0.25,0.3,0.35,0.4]
# get the ratio of case and control
CaseOverSamRatio = get_ratio(generate_path, FPR_max)
print(CaseOverSamRatio)

# get the combinations of the parameters
centerTimeinMins_list = [10, 30, 50, 60, 90]
optFea_WAOR_list = {'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'b': [6, 3, 2, 1.5, 1.2, 1, 6. / 7, 0.75, 6. / 9, 0.6]}
# centerTimeinMins_list = [50]
# optFea_WAOR_list = {
#     'a': [1],
#     'b': [6. / 9.]
# }
combination = []
for centerTimeinMins in centerTimeinMins_list:
    for alpha in optFea_WAOR_list['a']:
        for beta in optFea_WAOR_list['b']:
            combination.append([centerTimeinMins, alpha, beta])
print(len(combination))

trainortest = 'train'
for iscontrol in [0, 1]:
    if iscontrol == 0:
        caseorcontrol = 'case'
    else:
        caseorcontrol = 'control'
    hitstruct = np.load(
        tokenarray_path + caseorcontrol + '_' + trainortest + '_HitArray_dict_' + str(FPR_max)
        + '_sparse.npy', allow_pickle=True).item()
    patientid = list(hitstruct.keys())
    toolbox_input = np.load(
        tokenarray_path + caseorcontrol + '_' + trainortest + '_toolbox_input_' + str(FPR_max)
        + '_sparse.npy', allow_pickle=True)
    toolbox_input_df = pd.DataFrame({'PatID': toolbox_input[:, 0],
                                                'TimeToEnd': toolbox_input[:, 1],
                                                'HitOrNot': toolbox_input[:, 2]})
    print('Generating ' + trainortest + caseorcontrol + ' GWAOR ...')
    WAOR_pool(patient_list=patientid, trainortest=trainortest, caseorcontrol=caseorcontrol, iscontrol=iscontrol,
                          hitstruct=hitstruct, toolbox_input_df=toolbox_input_df, NumTrigCon=NumTrigCon,
                          CaseOverSamRatio=CaseOverSamRatio,
                            combination=combination, FPR_max=FPR_max, file_path=generate_path)

#
# get WAOR for testing set
NumTrigCon = 10
FPR_max = 0.25  # [0.02, 0.05, 0.1, 0.15, 0.2, 0.25,0.3,0.35,0.4]
CaseOverSamRatio = 0
# get the combinations of the parameters
centerTimeinMins_list = [10, 30, 50, 60, 90]
optFea_WAOR_list = {'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'b': [6, 3, 2, 1.5, 1.2, 1, 6. / 7, 0.75, 6. / 9, 0.6]}
combination = []
for centerTimeinMins in centerTimeinMins_list:
    for alpha in optFea_WAOR_list['a']:
        for beta in optFea_WAOR_list['b']:
            combination.append([centerTimeinMins, alpha, beta])
print(len(combination))

trainortest = 'test'
for iscontrol in [0, 1]:
    if iscontrol == 0:
        caseorcontrol = 'case'
    else:
        caseorcontrol = 'control'
    hitstruct = np.load(
        tokenarray_path + caseorcontrol + '_' + trainortest + '_HitArray_dict_' + str(FPR_max)
        + '_sparse.npy', allow_pickle=True).item()
    patientid = list(hitstruct.keys())
    toolbox_input = np.load(
        tokenarray_path + caseorcontrol + '_' + trainortest + '_toolbox_input_' + str(FPR_max)
        + '_sparse.npy', allow_pickle=True)
    toolbox_input_df = pd.DataFrame({'PatID': toolbox_input[:, 0],
                                                'TimeToEnd': toolbox_input[:, 1],
                                                'HitOrNot': toolbox_input[:, 2]})
    print('Generating ' + trainortest + caseorcontrol + ' GWAOR ...')
    WAOR_pool(patient_list=patientid, trainortest=trainortest, caseorcontrol=caseorcontrol, iscontrol=iscontrol,
                          hitstruct=hitstruct, toolbox_input_df=toolbox_input_df, NumTrigCon=NumTrigCon,
                          CaseOverSamRatio=CaseOverSamRatio,
                            combination=combination, FPR_max=FPR_max, file_path=generate_path)





