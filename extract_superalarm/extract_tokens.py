import numpy as np
import pandas as pd
import ast

FPR_max = 0.15
max_len = 10
test_case_num = 406
test_control_num = 221

superalarm_path = '/Users/winnwu/projects/Hu_Lab/COT_project/generate/superalarm/'
matrix_path = '/Users/winnwu/projects/Hu_Lab/COT_project/generate/tokens/matrix/'
pattern_number = 10
store_path = '/Users/winnwu/projects/Hu_Lab/COT_project/generate/extracted_alarms/'
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
fpr = list(np.array(control_hit) / test_control_num)
# filter out the patterns with length > max_len
index = [index for index, item in enumerate(superalarm) if max_len >= len(item) > 1]
superalarm = [superalarm[i] for i in index]
ppv = [ppv[i] for i in index]
tpr = [tpr[i] for i in index]

# create a list to store unique token names
unique_token = []
# create a list to store the source of each token
token_source = []

# get the top and bottom patterns based on ppv
# sort the patterns by ppv
index = np.argsort(ppv)[::-1]
superalarm_ppv = [superalarm[i] for i in index]
ppv_ppv = [ppv[i] for i in index]
tpr_ppv = [tpr[i] for i in index]
# assign each pattern with the corresponding token names
top_pattern_list_ppv = []
top_source_list_ppv = []
top_ppv_list_ppv = ppv_ppv[:pattern_number]
top_tpr_list_ppv = tpr_ppv[:pattern_number]
for i in range(pattern_number):
    pattern = superalarm_ppv[i]
    names = []
    for j in range(len(pattern)):
        subname = tokenid_loc[tokenid_loc['token_loc'] == pattern[j] - 1]['token'].values[0]
        if pattern[j] <= 78:
            names.append(subname + str(pattern[j] - 1))
        else:
            if pattern[j] >= 622 and pattern[j] <= 735:
                subname = (subname.replace('rr', 'repiratory rate').replace('VE', 'minute volume').
                           replace('PIP', 'peak inspiratory pressure').replace('insp_t', 'inspiratory time')
                           .replace('fio2', 'inspired O2 fraction'))
                subname = (subname.replace('_H', ' High').replace('_L', ' Low'))
            if pattern[j] >= 736:
                subname = (subname.replace('age65-', 'age > 65'))
                subname = 'Male' if subname == 'M' else subname
                subname = 'Female' if subname == 'F' else subname
            names.append(subname)
            # check if this token is already in the unique_token list
            # if not, add it to the list and record its source
            if subname not in unique_token:
                unique_token.append(subname)
                # assign the source of the token
                if pattern[j] <= 78:
                    token_source.append('vital signs')
                elif pattern[j] <= 621:
                    token_source.append('lab results')
                elif pattern[j] <= 735:
                    token_source.append('ventilation settings and measurements')
                else:
                    token_source.append('demographics')
    top_pattern_list_ppv.append(names)

bottom_pattern_list_ppv = []
bottom_source_list_ppv = []
bottom_ppv_list_ppv = ppv_ppv[-pattern_number:]
bottom_tpr_list_ppv = tpr_ppv[-pattern_number:]
for i in range(pattern_number):
    pattern = superalarm_ppv[-i - 1]
    names = []
    for j in range(len(pattern)):
        subname = tokenid_loc[tokenid_loc['token_loc'] == pattern[j] - 1]['token'].values[0]
        if pattern[j] <= 78:
            names.append(subname + str(pattern[j] - 1))
        else:
            if pattern[j] >= 622 and pattern[j] <= 735:
                subname = (subname.replace('rr', 'respiratory rate').replace('VE', 'minute volume').
                           replace('PIP', 'peak inspiratory pressure').replace('insp t', 'inspiratory time')
                           .replace('fio2', 'inspired O2 fraction'))
                subname = (subname.replace('_H', ' High').replace('_L', ' Low'))
            if pattern[j] >= 736:
                subname = (subname.replace('age65-', 'age > 65'))
                subname = subname.replace('OTHER', 'Other Races')
                subname = 'Male' if subname == 'M' else subname
                subname = 'Female' if subname == 'F' else subname
            names.append(subname)
            # check if this token is already in the unique_token list
            # if not, add it to the list and record its source
            if subname not in unique_token:
                print(subname.replace('age65-', 'age > 65'))
                unique_token.append(subname)
                # assign the source of the token
                if pattern[j] <= 78:
                    token_source.append('vital signs')
                elif pattern[j] <= 621:
                    token_source.append('lab results')
                elif pattern[j] <= 735:
                    token_source.append('ventilation settings and measurements')
                else:
                    token_source.append('demographics')
    bottom_pattern_list_ppv.append(names)

# make a dataframe to store the top and bottom patterns
top_pattern_df_ppv = pd.DataFrame({'pattern': top_pattern_list_ppv, 'ppv': top_ppv_list_ppv, 'tpr': top_tpr_list_ppv})
bottom_pattern_df_ppv = pd.DataFrame({'pattern': bottom_pattern_list_ppv, 'ppv': bottom_ppv_list_ppv, 'tpr': bottom_tpr_list_ppv})
top_pattern_df_ppv.to_csv(store_path + 'top_patterns_ppv.csv', index=False)
bottom_pattern_df_ppv.to_csv(store_path + 'bottom_patterns_ppv.csv', index=False)


# get the top and bottom patterns based on tpr
# sort the patterns by tpr
index = np.argsort(tpr)[::-1]
superalarm_tpr = [superalarm[i] for i in index]
ppv_tpr = [ppv[i] for i in index]
tpr_tpr = [tpr[i] for i in index]
# assign each pattern with the corresponding token names
top_pattern_list_tpr = []
top_source_list_tpr = []
top_ppv_list_tpr = ppv_tpr[:pattern_number]
top_tpr_list_tpr = tpr_tpr[:pattern_number]
for i in range(pattern_number):
    pattern = superalarm_tpr[i]
    names = []
    for j in range(len(pattern)):
        subname = tokenid_loc[tokenid_loc['token_loc'] == pattern[j] - 1]['token'].values[0]
        if pattern[j] <= 78:
            names.append(subname + str(pattern[j] - 1))
        else:
            if pattern[j] >= 622 and pattern[j] <= 735:
                subname = (subname.replace('rr', 'respiratory rate').replace('VE', 'minute volume').
                           replace('PIP', 'peak inspiratory pressure').replace('insp t', 'inspiratory time')
                           .replace('fio2', 'inspired O2 fraction'))
                subname = (subname.replace('_H', ' High').replace('_L', ' Low'))
            if pattern[j] >= 736:
                subname = (subname.replace('age65-', 'age > 65'))
                subname = subname.replace('OTHER', 'Other Races')
                subname = 'Male' if subname == 'M' else subname
                subname = 'Female' if subname == 'F' else subname
            names.append(subname)
        if subname not in unique_token:
            unique_token.append(subname)
            # assign the source of the token
            if pattern[j] <= 78:
                token_source.append('vital signs')
            elif pattern[j] <= 621:
                token_source.append('lab results')
            elif pattern[j] <= 735:
                token_source.append('ventilation settings and measurements')
            else:
                token_source.append('demographics')
    top_pattern_list_tpr.append(names)

bottom_pattern_list_tpr = []
bottom_source_list_tpr = []
bottom_ppv_list_tpr = ppv_tpr[-pattern_number:]
bottom_tpr_list_tpr = tpr_tpr[-pattern_number:]
for i in range(pattern_number):
    pattern = superalarm_tpr[-i - 1]
    names = []
    for j in range(len(pattern)):
        subname = tokenid_loc[tokenid_loc['token_loc'] == pattern[j] - 1]['token'].values[0]
        if pattern[j] <= 78:
            names.append(subname + str(pattern[j] - 1))
        else:
            if pattern[j] >= 622 and pattern[j] <= 735:
                subname = (subname.replace('rr', 'respiratory rate').replace('VE', 'minute volume').
                           replace('PIP', 'peak inspiratory pressure').replace('insp t', 'inspiratory time')
                           .replace('fio2', 'inspired O2 fraction'))
                subname = (subname.replace('_H', ' High').replace('_L', ' Low'))
            if pattern[j] >= 736:
                subname = (subname.replace('age65-', 'age > 65'))
                subname = subname.replace('OTHER', 'Other Races')
                subname = 'Male' if subname == 'M' else subname
                subname = 'Female' if subname == 'F' else subname
            names.append(subname)
        if subname not in unique_token:
            unique_token.append(subname)
            # assign the source of the token
            if pattern[j] <= 78:
                token_source.append('vital signs')
            elif pattern[j] <= 621:
                token_source.append('lab results')
            elif pattern[j] <= 735:
                token_source.append('ventilation settings and measurements')
            else:
                token_source.append('demographics')
    bottom_pattern_list_tpr.append(names)

# make a dataframe to store the top and bottom patterns
top_pattern_df_tpr = pd.DataFrame({'pattern': top_pattern_list_tpr, 'ppv': top_ppv_list_tpr, 'tpr': top_tpr_list_tpr})
bottom_pattern_df_tpr = pd.DataFrame({'pattern': bottom_pattern_list_tpr, 'ppv': bottom_ppv_list_tpr, 'tpr': bottom_tpr_list_tpr})
top_pattern_df_tpr.to_csv(store_path + 'top_patterns_tpr.csv', index=False)
bottom_pattern_df_tpr.to_csv(store_path + 'bottom_patterns_tpr.csv', index=False)



#make another dataframe to store the unique token names and their sources
token_df = pd.DataFrame({'token': unique_token, 'source': token_source, 'description': None, 'reference': None})
token_df.to_csv(store_path + 'unique_tokens.csv', index=False)