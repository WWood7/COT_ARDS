import numpy as np

tokenarray_path = '/Users/winnwu/projects/Hu_Lab/COT_project/generate/tokenarray'
maxFPR = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

ratio = []
for FPR in maxFPR:
    case_data = np.load(tokenarray_path + '/case_train_toolbox_input_' + str(FPR) + '_sparse.npy', allow_pickle=True)
    control_data = np.load(tokenarray_path + '/control_train_toolbox_input_' + str(FPR) + '_sparse.npy', allow_pickle=True)
    case_triggers_count = np.sum(case_data[:, 2])
    control_triggers_count = np.sum(control_data[:, 2])
    ratio.append(control_triggers_count / case_triggers_count)
    average_case_triggers = np.sum(case_data[:, 2]) / len(np.unique(case_data[:, 0]))
    average_control_triggers = np.sum(control_data[:, 2]) / len(np.unique(control_data[:, 0]))
    print('ratio:', ratio)
    print('average_case_triggers:', average_case_triggers)
    print('average_control_triggers:', average_control_triggers)
