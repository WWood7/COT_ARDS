import numpy as np
import matplotlib.pyplot as plt
import ast

file_path = '/Users/winnwu/projects/emory-hu lab/workshop/generate/'
# read in the hit array
case_test = np.load(file_path + 'tokenarray/case_test_HitArray_dict_0.05_sparse.npy', allow_pickle=True).item()
case_test_keys = list(case_test.keys())
control_test = np.load(file_path + 'tokenarray/control_test_HitArray_dict_0.05_sparse.npy', allow_pickle=True).item()
control_test_keys = list(control_test.keys())

# for case test patients, 50, 80, 105, 70, 23, 27, 28
# for control test patients
index = 30
case_test_hit = case_test[case_test_keys[index]]['sparseHitArray']
case_test_time = case_test[case_test_keys[index]]['HitT']
case_test_dense_hit = case_test_hit.toarray()
case_test_time = case_test_time - np.max(case_test_time)

# # get the positions for eventplots
# case_test_pos = []
# for i in range(case_test_hit.shape[0]):
#     case_test_subpos = []
#     for j in range(case_test_hit.shape[1]):
#         if case_test_hit[i, j] == 1:
#             case_test_subpos.append(case_test_time[j])
#     case_test_pos.append(np.array(case_test_subpos))
# plt.eventplot(case_test_pos, color='black', linelengths=2)
# plt.xlabel('Relative Time to Onset (hours)')
# plt.ylabel('COT sets')
# plt.show()


for i in range(70, 100, 1):
    control_test_hit = control_test[control_test_keys[i]]['sparseHitArray']
    control_test_dense_hit = control_test_hit.toarray()
    print(np.shape(control_test_dense_hit))
    if np.sum(control_test_dense_hit) > 0:
        control_test_time = control_test[control_test_keys[i]]['HitT']
        control_test_time = control_test_time - np.max(control_test_time)

        control_test_pos = []
        for k in range(control_test_hit.shape[0]):
            control_test_subpos = []
            for j in range(control_test_hit.shape[1]):
                if control_test_hit[k, j] == 1:
                    control_test_subpos.append(control_test_time[j])
            control_test_pos.append(np.array(control_test_subpos))
        plt.eventplot(control_test_pos, color='black', linelengths=2)
        plt.xlabel('Relative Time to End of Segment (hours)')
        plt.ylabel('COT sets')
        print(i)
        plt.show()
        break






