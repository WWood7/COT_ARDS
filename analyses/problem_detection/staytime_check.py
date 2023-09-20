import numpy as np
import pandas as pd

mimiciii_path = '/Users/winnwu/projects/Hu_Lab/COT_project/generate/tokenarray/'
mimiciv_path = '/Users/winnwu/projects/Hu_Lab/COT_project/generate/mimiciv/tokenarray/'

# get the occurrence time of the first clinical event
# get the total number of clinical events
def get_occurrence(data):
    first_occurrence_time = []
    occurrence_number = []
    id = np.unique(data[:, 0])
    for i in id:
        sub_data = data[data[:, 0] == i]
        first_occurrence_time.append(sub_data[0, 1])
        occurrence_number.append(sub_data.shape[0])
    return [first_occurrence_time, occurrence_number]

# FPR_max does not matter here
mimiciii_case = np.load(mimiciii_path + 'case_test_toolbox_input_0.3_sparse.npy', allow_pickle=True)
mimiciii_control = np.load(mimiciii_path + 'control_test_toolbox_input_0.3_sparse.npy', allow_pickle=True)
mimiciv_case = np.load(mimiciv_path + 'case_toolbox_input_0.3_sparse.npy', allow_pickle=True)
mimiciv_control = np.load(mimiciv_path + 'control_toolbox_input_0.3_sparse.npy', allow_pickle=True)
mimiciv_case2 = np.load(mimiciv_path + 'case_toolbox_input_0.05_sparse.npy', allow_pickle=True)

mimiciii_case_occurrence = get_occurrence(mimiciii_case)
mimiciii_control_occurrence = get_occurrence(mimiciii_control)
mimiciv_case_occurrence = get_occurrence(mimiciv_case)
mimiciv_control_occurrence = get_occurrence(mimiciv_control)

print('mimiciii_case_occurrence: ', np.mean(mimiciii_case_occurrence[1]), np.std(mimiciii_case_occurrence[1]))
print('mimiciii_case_time:', np.mean(mimiciii_case_occurrence[0]), np.std(mimiciii_case_occurrence[0]))
print('mimiciii_control_occurrence: ', np.mean(mimiciii_control_occurrence[1]), np.std(mimiciii_control_occurrence[1]))
print('mimiciii_control_time:', np.mean(mimiciii_control_occurrence[0]), np.std(mimiciii_control_occurrence[0]))
print('mimiciv_case_occurrence: ', np.mean(mimiciv_case_occurrence[1]), np.std(mimiciv_case_occurrence[1]))
print('mimiciv_case_time:', np.mean(mimiciv_case_occurrence[0]), np.std(mimiciv_case_occurrence[0]))
print('mimiciv_control_occurrence: ', np.mean(mimiciv_control_occurrence[1]), np.std(mimiciv_control_occurrence[1]))
print('mimiciv_control_time:', np.mean(mimiciv_control_occurrence[0]), np.std(mimiciv_control_occurrence[0]))