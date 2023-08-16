import numpy as np
import pandas as pd
import ast

FPR_max = 0.15
max_len = 9
test_case_num = 406
test_control_num = 221
superalarm_path = '/Users/winnwu/projects/Hu_lab/COT_project/generate/superalarm/'
matrix_path = '/Users/winnwu/projects/Hu_lab/COT_project/generate/tokens/matrix/'
pattern_number = 10
store_path = '/Users/winnwu/projects/Hu_lab/COT_project/generate/extracted_alarms/'

# read in the token names
tokenid_loc = pd.read_csv(matrix_path + 'tokenid_loc.csv')

# read in the pattern set and hit counts
with open(superalarm_path + 'superalarm_' + str(FPR_max) + '.txt', "r") as file:
    content = file.read()
    superalarm = ast.literal_eval(content)
with open(superalarm_path + 'TP4EachPattern_' + str(FPR_max) + '.txt', "r") as file:
    content = file.read()
    case_hit = ast.literal_eval(content)
with open(superalarm_path + 'FP4EachPattern_' + str(FPR_max) + '.txt', "r") as file:
    content = file.read()
    control_hit = ast.literal_eval(content)
ppv = list(np.array(case_hit) / (np.array(case_hit) + np.array(control_hit)))
tpr = list(np.array(case_hit) / test_case_num)
# filter out the patterns with length > max_len
index = [index for index, item in enumerate(superalarm) if len(item) <= max_len and len(item) > 1]
superalarm = [superalarm[i] for i in index]
ppv = [ppv[i] for i in index]
tpr = [tpr[i] for i in index]
print(len(superalarm))
# sort the patterns by ppv
index = np.argsort(tpr)[::-1]
superalarm = [superalarm[i] for i in index]
ppv = [ppv[i] for i in index]
tpr = [tpr[i] for i in index]
# assign each pattern with the corresponding token names
top_pattern = {}
for i in range(pattern_number):
    pattern = superalarm[i]
    names = ''
    for j in range(len(pattern)):
        if pattern[j] <= 78:
            names = (names + tokenid_loc[tokenid_loc['token_loc'] == pattern[j] - 1]['token'].values[0] +
                     str(pattern[j] - 1) + '; ')
        else:
            names = names + tokenid_loc[tokenid_loc['token_loc'] == pattern[j] - 1]['token'].values[0] + '; '
    top_pattern[names] = ppv[i]
bottom_pattern = {}
for i in range(pattern_number):
    pattern = superalarm[-i - 1]
    names = ''
    for j in range(len(pattern)):
        if pattern[j] <= 78:
            names = (names + tokenid_loc[tokenid_loc['token_loc'] == pattern[j] - 1]['token'].values[0] +
                     str(pattern[j] - 1) + '; ')
        else:
            names = names + tokenid_loc[tokenid_loc['token_loc'] == pattern[j] - 1]['token'].values[0] + '; '
    bottom_pattern[names] = ppv[-i-1]
# save the results
with open(store_path + 'top_pattern_' + str(FPR_max) + '.txt', 'w') as file:
    file.write(str(top_pattern))
with open(store_path + 'bottom_pattern_' + str(FPR_max) + '.txt', 'w') as file:
    file.write(str(bottom_pattern))

