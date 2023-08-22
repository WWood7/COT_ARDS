import numpy as np
import pandas as pd
import ast
import json

FPR_max = 0.15
max_len = 20
test_case_num = 406
test_control_num = 221
superalarm_path = '/Users/winnwu/projects/Hu_lab/COT_project/generate/superalarm/'
matrix_path = '/Users/winnwu/projects/Hu_lab/COT_project/generate/tokens/matrix/'
pattern_number = 5
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
index = np.argsort(ppv)[::-1]
superalarm = [superalarm[i] for i in index]
ppv = [ppv[i] for i in index]
tpr = [tpr[i] for i in index]
# assign each pattern with the corresponding token names
top_pattern_list = []
top_source_list = []
top_ppv_list = ppv[:pattern_number]
for i in range(pattern_number):
    pattern = superalarm[i]
    names = []
    sources = []
    for j in range(len(pattern)):
        subname = tokenid_loc[tokenid_loc['token_loc'] == pattern[j] - 1]['token'].values[0]
        if pattern[j] <= 78:
            names.append(subname + str(pattern[j] - 1))
        else:
            if pattern[j] >= 622 and pattern[j] <= 735:
                subname = (subname.replace('rr', 'repiratory_rate').replace('VE', 'minute_volume').
                           replace('PIP', 'peak_insp_pressure').replace('insp_t', 'inspiratory_time')
                           .replace('fio2', 'inspired_O2_fraction'))
            names.append(subname)
        if pattern[j] <= 78:
            sources.append('vital_signs')
        elif pattern[j] <= 621:
            sources.append('lab_results')
        elif pattern[j] <= 735:
            sources.append('ventilation_settings_and_measurements')
        else:
            sources.append('demographics')
    top_pattern_list.append(names)
    top_source_list.append(sources)


bottom_pattern_list = []
bottom_source_list = []
bottom_ppv_list = ppv[-pattern_number:]
for i in range(pattern_number):
    pattern = superalarm[-i - 1]
    names = []
    sources = []
    for j in range(len(pattern)):
        subname = tokenid_loc[tokenid_loc['token_loc'] == pattern[j] - 1]['token'].values[0]
        if pattern[j] <= 78:
            names.append(subname + str(pattern[j] - 1))
        else:
            if pattern[j] >= 622 and pattern[j] <= 735:
                subname = (subname.replace('rr', 'repiratory_rate').replace('VE', 'minute_volume').
                           replace('PIP', 'peak_insp_pressure').replace('insp_t', 'inspiratory_time')
                           .replace('fio2', 'inspired_O2_fraction'))
            names.append(subname)
        if pattern[j] <= 78:
            sources.append('vital_signs')
        elif pattern[j] <= 621:
            sources.append('lab_results')
        elif pattern[j] <= 735:
            sources.append('ventilation_settings_and_measurements')
        else:
            sources.append('demographics')
    bottom_pattern_list.append(names)
    bottom_source_list.append(sources)


# save the results as JSON
def combine_and_save_as_json(tokens_list, source_list, filename, ppv_list):
    # 确保tokens_list和source_list长度相同
    print(len(tokens_list), len(source_list), len(ppv_list))
    assert len(tokens_list) == len(source_list), "Length of tokens_list and source_list should be same."
    assert len(tokens_list) == len(ppv_list), "Length of tokens_list and ppv_list should be same."

    combined_data = []
    for tokens, source, ppv in zip(tokens_list, source_list, ppv_list):
        combined_data.append({
            "tokens": tokens,
            "source": source,
            'ppv': ppv
        })

    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(combined_data, json_file, ensure_ascii=False, indent=4)

# 使用函数保存数据
combine_and_save_as_json(top_pattern_list, top_source_list, store_path + 'top_patterns.json', top_ppv_list)
combine_and_save_as_json(bottom_pattern_list, bottom_source_list, store_path + 'bottom_patterns.json', bottom_ppv_list)

