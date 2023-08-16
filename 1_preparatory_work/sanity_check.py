import pandas as pd

file_path = '/Users/winnwu/projects/Hu_lab/COT_project/generate/segments/'
train_case_segs = pd.read_csv(file_path + 'train_case_segs.csv')
train_control_segs = pd.read_csv(file_path + 'train_control_segs.csv')
test_case_segs = pd.read_csv(file_path + 'test_case_segs.csv')
test_control_segs = pd.read_csv(file_path + 'test_control_segs.csv')
print('# of train case:' + str(len(train_case_segs)))
print('# of train control:' + str(len(train_control_segs)))
print('# of test case:' + str(len(test_case_segs)))
print('# of test control:' + str(len(test_control_segs)))