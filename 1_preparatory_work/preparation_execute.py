import os
from sklearn import model_selection
from preparation_function import *

#
# define all the parameters
test_ratio = 0.2
window_length = 12
seed = 7
data_path = '/Users/winnwu/projects/emory-hu lab/COT_project/data/'
generate_path = '/Users/winnwu/projects/emory-hu lab/COT_project/generate/'

#
# find patient cohort
case_patients = pd.read_csv(data_path + 'LLARDS_100.csv')
case_patients = case_patients.rename(columns={"charttime": "onset"})
control_patients = pd.read_csv(data_path + 'LLNONARDS_300_WODIE.csv')

#
# split the patients into training set and test set by subject_id
# case patients don't have repeated subject_id, so we can directly split the subject_id
# control patients have repeated subject_id, so split by unique subject_id
# but very few control patients have repeated subject_id, so it does not affect our ratio of split
train_case_sub, test_case_sub = \
    model_selection.train_test_split(case_patients['subject_id'].unique(), test_size=test_ratio, random_state=seed)
train_control_sub, test_control_sub = \
    model_selection.train_test_split(control_patients['subject_id'].unique(), test_size=test_ratio, random_state=seed)
train_case, test_case = case_patients[case_patients['subject_id'].isin(train_case_sub)], \
    case_patients[case_patients['subject_id'].isin(test_case_sub)]
train_control, test_control = control_patients[control_patients['subject_id'].isin(train_control_sub)], \
    control_patients[control_patients['subject_id'].isin(test_control_sub)]
training_patients = {'case': train_case, 'control': train_control}
test_patients = {'case': test_case, 'control': test_control}

#
# form and save the segments
train_segs, test_segs = form_segments(window_length, training_patients, test_patients, seed)

folder_path = generate_path + 'segments'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

train_segs['case'].to_csv(folder_path + '/train_case_segs.csv')
train_segs['control'].to_csv(folder_path + '/train_control_segs.csv')
test_segs['case'].to_csv(folder_path + '/test_case_segs.csv')
test_segs['control'].to_csv(folder_path + '/test_control_segs.csv')
