import pandas as pd
import random
import datetime


def form_segments(length, train, test, seed):
    random.seed(seed)
    #
    # for case patients,
    # segments are just the window with specified length preceding onset
    train_case_seg, test_case_seg = train['case'].copy(), test['case'].copy()

    train_case_seg['segstart'] = pd.to_datetime(train_case_seg['onset']) + \
                                 datetime.timedelta(hours=-length)
    train_case_seg['segend'] = pd.to_datetime(train_case_seg['onset'])
    train_case_seg['seg_id'] = train_case_seg['icustay_id']
    train_case_seg.reset_index(drop=True)
    test_case_seg['segstart'] = pd.to_datetime(test_case_seg['onset']) + \
                                datetime.timedelta(hours=-length)
    test_case_seg['segend'] = pd.to_datetime(test_case_seg['onset'])
    test_case_seg['seg_id'] = test_case_seg['icustay_id']
    test_case_seg.reset_index(drop=True)

    #
    # for control patients,
    # for each unique icustay, if there are multiple ventilation periods recorded,
    # choose a random one
    # the segment will be the window with specified length preceding conend
    # (the end time of that ventilation period?)
    train_control_one = []
    for pat in train['control']['icustay_id'].unique():
        sub_train_control = train['control'][
            train['control']['icustay_id'] == pat].reset_index(drop=True)
        if len(sub_train_control) != 0:
            sub_train_control = sub_train_control[
                sub_train_control.index == random.randint(0, len(sub_train_control) - 1)]
            train_control_one.append(sub_train_control)
    train_control = pd.concat(train_control_one)

    test_control_one = []
    for pat in test['control']['icustay_id'].unique():
        sub_test_control = test['control'][
            test['control']['icustay_id'] == pat].reset_index(drop=True)
        if len(sub_test_control) != 0:
            sub_test_control = sub_test_control[
                sub_test_control.index == random.randint(0, len(sub_test_control) - 1)]
            test_control_one.append(sub_test_control)
    test_control = pd.concat(test_control_one)

    train_control_seg, test_control_seg = train_control.copy(), test_control.copy()
    train_control_seg['segstart'] = pd.to_datetime(train_control_seg['conend']) + \
                                    datetime.timedelta(hours=-length)
    train_control_seg['segend'] = pd.to_datetime(train_control_seg['conend'])
    train_control_seg['seg_id'] = train_control_seg['icustay_id']
    train_control_seg.reset_index(drop=True)
    test_control_seg['segstart'] = pd.to_datetime(test_control_seg['conend']) + \
                                   datetime.timedelta(hours=-length)
    test_control_seg['segend'] = pd.to_datetime(test_control_seg['conend'])
    test_control_seg['seg_id'] = test_control_seg['icustay_id']
    test_control_seg.reset_index(drop=True)

    train_segs = {'case': train_case_seg, 'control': train_control_seg}
    test_segs = {'case': test_case_seg, 'control': test_control_seg}

    return train_segs, test_segs
