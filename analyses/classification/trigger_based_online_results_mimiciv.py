import numpy as np

tokenarray_path = '/Users/winnwu/projects/Hu_Lab/COT_project/generate/mimiciv/tokenarray/'
for FPR_max in [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
     test_case = np.load(tokenarray_path + 'case_toolbox_input_' + str(FPR_max) + '_sparse.npy',
                         allow_pickle=True)

     test_control = np.load(tokenarray_path + 'control_toolbox_input_' + str(FPR_max) + '_sparse.npy',
                            allow_pickle=True)
     n_case = len(np.unique(test_case[:, 0]))
     n_control = len(np.unique(test_control[:, 0]))
     trigger_case = 0
     for i in np.unique(test_case[:, 0]):
         trigger = sum(test_case[test_case[:, 0] == i, 2])
         if trigger > 0:
             trigger_case += 1
     trigger_control = 0
     for i in np.unique(test_control[:, 0]):
         trigger = sum(test_control[test_control[:, 0] == i, 2])
         if trigger > 0:
             trigger_control += 1

     FPR = trigger_control / n_control
     TPR = trigger_case / n_case
     precision = trigger_case / (trigger_case + trigger_control)
     F1 = 2 * precision * TPR / (precision + TPR)
     print(str(FPR_max))
     print('TPR: ' + str(TPR))
     print('FPR: ' + str(FPR))
     print('precision: ' + str(precision))
     print('F1: ' + str(F1))


