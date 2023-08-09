import numpy as np


case = np.load('/Users/winnwu/projects/Hu_Lab/COT_project/generate/tokenarray/'
                   'case_train_toolbox_input_0.4_sparse.npy', allow_pickle=True)
control = np.load('/Users/winnwu/projects/Hu_Lab/COT_project/generate/tokenarray/'
                        'control_train_toolbox_input_0.4_sparse.npy', allow_pickle=True)
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

print(enc_case)
print(enc_control)
print(enc_control / enc_case)
print(len(np.unique(control[:, 0])))