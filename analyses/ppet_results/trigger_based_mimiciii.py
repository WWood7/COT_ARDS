from core import scorers, Process, augmenters, utils, mews,run
import numpy as np
from matplotlib import pyplot as plt

path = '/Users/winnwu/projects/Hu_Lab/COT_project/generate/tokenarray/'
FPR_max = 0.15

# read in the data
case = np.load(path + 'case_test_toolbox_input_' + str(FPR_max) + '_sparse.npy', allow_pickle=True)
control = np.load(path + 'control_test_toolbox_input_' + str(FPR_max) + '_sparse.npy', allow_pickle=True)

thresh = [1]
case_scorers = [scorers.PosNeg(tmin=0, tmax=12)]
augmenter = augmenters.NoAugment()
# a class to integrate all parameters together
case_processor = Process(thresholds=thresh, scorers=case_scorers, augmenter=augmenter).per_data
case_count, case_count_raw = run(case, case_processor)
control_scorers = [scorers.PosNeg(tmin=0, tmax=np.inf)]
# 100 random selection of time windows (12 hours) from each patient record, sort these windown (if choose not to sort, assign 0 to the is_sort flag) by timeline
augmenter = augmenters.RandomWindows(num_samples=100, duration=12, is_sort=1)
control_processor = Process(thresholds=thresh, scorers=control_scorers, augmenter=augmenter).per_data
control_count, control_count_raw = run(control, control_processor)
i = 0
TP = case_count[0][:, i, 0]
FN = case_count[0][:, i, 1]
FP = control_count[0][:, i, 0]
TN = control_count[0][:, i, 1]
total_positives = TP[0] + FN[0]
total_negatives = FP[0] + TN[0]

# Sensitivity
TPR = TP / (total_positives)
# Specificity
TNR = TN / (total_negatives)
# Precision or positive predictive value
PPV = TP / (TP + FP)
# work up to detection ratio
WDR = 1 / PPV
# Negative predictive value
NPV = TN / (TN + FN)
# Fall out or false positive rate
FPR = FP / (total_negatives)
# False negative rate
FNR = FN / (total_positives)
# accuracy
ACC = (TP + TN) / (total_negatives + total_positives)  # RanXiao: add ACC and F1 for
# F1 scores
F1 = 2 * (PPV * TPR) / (PPV + TPR)

print('Performance metrics with threshold ' + str(thresh[i]) + ':')
print('TPR', 'FPR', 'PPV', 'NPV', 'TNR', 'FNR', 'WDR', 'ACC', 'F1')
print('Mean')
print(np.array([np.mean(x) for x in [TPR, FPR, PPV, NPV, TNR, FNR, WDR, ACC, F1]]).T)
print('Std')
print(np.array([np.std(x) for x in [TPR, FPR, PPV, NPV, TNR, FNR, WDR, ACC, F1]]).T)
print('\n')


thresh = [1] #define threshold or thresholds to be used
# Define scorer for calculating false positive for case and control separately. for case it's defined as duration outside of prediction horizon, for control it's the whole duration
case_scorers = [scorers.ProportionWarning_case(tlead=0, twin=12)] # 0h lead time and 12h prediction horizon
control_scorers = [scorers.ProportionWarning(tmin=0, tmax=np.inf)]# whole control data
# no augmentation is needed for calculation of false alarm proportion
augmenter = augmenters.NoAugment()
# generate FAP for case condition
case_processor = Process(thresholds=thresh, scorers=case_scorers, augmenter=augmenter).per_data
case_count, case_count_raw = run(case, case_processor)
# generate FAP control condition
control_processor = Process(thresholds=thresh, scorers=control_scorers, augmenter=augmenter).per_data
control_count, control_count_raw = run(control, control_processor)
# calculate FAP
FAP_case_mean = np.nanmean(np.squeeze(case_count_raw), axis=0)
FAP_control_mean = np.nanmean(np.squeeze(control_count_raw), axis=0)
FAP_conbined_mean = np.nanmean(np.squeeze(np.append(case_count_raw[0], control_count_raw[0], axis=0)),axis=0)
FAP_case_std = np.nanstd(np.squeeze(case_count_raw), axis=0)
FAP_control_std = np.nanstd(np.squeeze(control_count_raw), axis=0)
FAP_conbined_std = np.nanstd(np.squeeze(np.append(case_count_raw[0], control_count_raw[0], axis=0)),axis=0)
print('FAP_case_mean:', FAP_case_mean)
print('FAP_control_mean:', FAP_control_mean)
print('FAP_conbined_mean:', FAP_conbined_mean)



# calculate FAR
thresh = [1]#define threshold or thresholds to be used
# Define scorer for calculating false positive for case and control separately. for case it's defined as duration outside of prediction horizon, for control it's the whole duration
case_scorers = [scorers.HourlyFalseAlarmRate_case(tlead=0, twin=12)] # 0h lead time and 12h prediction horizon
control_scorers = [scorers.HourlyFalseAlarmRate(tmin=0, tmax=np.inf)] # whole control data
# no augmentation is needed for calculation of false alarm rate
augmenter = augmenters.NoAugment()
# generate FAR for case condition
case_processor = Process(thresholds=thresh, scorers=case_scorers, augmenter=augmenter).per_data
case_count, case_count_raw = run(case, case_processor)
# generate FAR for control condition
control_processor = Process(thresholds=thresh, scorers=control_scorers, augmenter=augmenter).per_data
control_count, control_count_raw = run(control, control_processor)
# calcualte mean and std fo false alarm rate for case, control and their combination
FAR_case_mean = np.nanmean(np.squeeze(case_count_raw),axis=0)
FAR_control_mean = np.nanmean(np.squeeze(control_count_raw),axis=0)
FAR_conbined_mean = np.nanmean(np.squeeze(np.append(case_count_raw[0],control_count_raw[0],axis=0)),axis=0)
FAR_case_std = np.nanstd(np.squeeze(case_count_raw),axis=0)
FAR_control_std = np.nanstd(np.squeeze(control_count_raw),axis=0)
FAR_conbined_std = np.nanstd(np.squeeze(np.append(case_count_raw[0],control_count_raw[0],axis=0)),axis=0)
print('FAR_case_mean:', FAR_case_mean)
print('FAR_control_mean:', FAR_control_mean)
print('FAR_conbined_mean:', FAR_conbined_mean)