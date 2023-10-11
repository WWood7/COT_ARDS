from core import scorers, Process, augmenters, utils, mews, run
import numpy as np
import itertools

FPR_max = 0.15
path = '/Users/winnwu/projects/Hu_Lab/COT_project/generate/XGB_results/mimiciii/'
case = np.load(path + 'test_case_seq_toolbox_input.npy', allow_pickle=True)
another_path = '/Users/winnwu/projects/Hu_Lab/COT_project/generate/tokenarray/'
case_trigger = np.load(another_path + 'case_test_toolbox_input_' + str(FPR_max) + '_sparse.npy', allow_pickle=True)
ids = np.unique(case_trigger[:, 0])
# get the first 10 ids
ids = ids[:10]
# get the corresponding case data
case_trigger = case_trigger[np.isin(case_trigger[:, 0], ids)]


thresh = [0.8]
case_scorers = [scorers.PosNeg(tmin=0, tmax=12)]
augmenter = augmenters.NoAugment()
# a class to integrate all parameters together
case_processor = Process(thresholds=thresh, scorers=case_scorers, augmenter=augmenter).per_data
case_count, case_count_raw = run(case, case_processor)

i = 0
TP = case_count[0][:, i, 0]
FN = case_count[0][:, i, 1]
total_positives = TP[0] + FN[0]

# Sensitivity
TPR = TP / (total_positives)
print(TPR)
print(case.shape)
print(case_trigger.shape)