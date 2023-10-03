import numpy as np

from core import mews
from core import scorers, Process, run
from core import augmenters


def sensitivity_lead(case, thresh, lead_times, discounted=True):
    scorer_functions = [scorers.Lead(lead_times=lead_times)]
    augmenter = augmenters.NoAugment()
    processor = Process(thresholds=thresh, scorers=scorer_functions, augmenter=augmenter).per_data

    case_count, case_count_raw = run(case, processor)
    cur_case = case_count[0]
    if discounted:
        sensitivity = cur_case[0, :, :, 0] / cur_case[0, :, :, 1]
    else:
        sensitivity = cur_case[0, :, :, 0] / len(np.unique(case[:, 0]))
    return sensitivity


def all_metrics(case, control, thresh, tmax=np.inf):
    scorer_functions = [scorers.PosNeg(tmin=0, tmax=tmax)]
    augmenter = augmenters.NoAugment()
    processor = Process(thresholds=thresh, scorers=scorer_functions, augmenter=augmenter).per_data

    case_count, case_count_raw = run(case, processor)
    control_count, control_count_raw = run(control, processor)

    cur_case = np.transpose(case_count[0], (2, 1, 0))
    cur_control = np.transpose(control_count[0], (2, 1, 0))

    TP = cur_case[0].reshape(-1,);  FN = cur_case[1].reshape(-1,)
    FP = cur_control[0].reshape(-1,);  TN = cur_control[1].reshape(-1,)

    total_positives = len(np.unique(case[:, 0]))
    total_negatives = len(np.unique(control[:, 0]))
    # Sensitivity or true negative rate
    TPR = TP / (total_positives)
    # Specificity or true negative rate
    TNR = TN / (total_negatives)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # WDR
    WDR = 1 / PPV
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (total_negatives)
    # False negative rate
    FNR = FN / (total_positives)
    return {'TPR': TPR, 'FPR': FPR, 'FNR': FNR, 'TNR':  TNR, 'NPV': NPV, 'PPV':  PPV,'WDR':  WDR}

def roc(case, control, thresh, tmax=np.inf):
    scorer_functions = [scorers.PosNeg(tmin=0, tmax=tmax)]
    augmenter = augmenters.NoAugment()
    processor = Process(thresholds=thresh, scorers=scorer_functions, augmenter=augmenter).per_data

    case_count, case_count_raw = run(case, processor)
    control_count, control_total, control_count_raw, control_count_std = run(control, processor)

    cur_case = np.transpose(case_count[0], (2, 1, 0))
    cur_control = np.transpose(control_count[0], (2, 1, 0))

    TP = cur_case[0].reshape(-1,)
    FP = cur_control[0].reshape(-1,)

    total_positives = len(np.unique(case[:, 0]))
    total_negatives = len(np.unique(control[:, 0]))
    # Sensitivity or true negative rate
    TPR = TP / (total_positives)
    FPR = FP / (total_negatives)

    return FPR, TPR

if __name__ == '__main__':
    case, _ = mews.prepare_case_multiple()
    control, _ = mews.prepare_control()

    print(all_metrics(case, control, np.arange(0,14)))
    print(sensitivity_lead(case, thresh=np.arange(0,14), lead_times=np.arange(0, 12, 10/60), discounted=False))

