import pandas as pd
import json
import ast

path = '/Users/winnwu/projects/Hu_lab/COT_project/generate/extracted_alarms/'
ppv_store_path = path + 'tokens_ppv/'
sensitivity_store_path = path + 'tokens_sensitivity/'

# read in token_info and token lists
tokens_info = pd.read_csv(path + 'tokens_info.csv')

# define a function to make tokens into json files
def make_json(cot, ids, store_path, type):
    for i in range(len(cot)):
        temp_cot = ast.literal_eval(cot['pattern'][i])
        tokens_list = []
        for j in range(len(temp_cot)):
            token = temp_cot[j]
            token_info_row = tokens_info[tokens_info['token'] == token]
            if token_info_row.empty:
                print('token not found:', token)
                continue
            source = token_info_row['source'].values[0] if not token_info_row[
                'source'].isnull().values.any() else ' '
            description = token_info_row['description'].values[0] if not token_info_row[
                'description'].isnull().values.any() else ' '
            referenceRange = token_info_row['reference'].values[0] if not token_info_row[
                'reference'].isnull().values.any() else ' '
            tokens_list.append({
                'token': token,
                'source': source,
                'description': description,
                'referenceRange': referenceRange
            })
        final_json = {
            'tokens': tokens_list,
            'ppv': cot['ppv'][i],
            'sensitivity': cot['tpr'][i],
            'type': type
        }
        with open(store_path + str(ids[i]) + '.json', 'w', encoding='utf-8') as json_file:
            json.dump(final_json, json_file, ensure_ascii=False, indent=4)


# make top ppv json files
top_ppv_cot = pd.read_csv(path + 'top_patterns_ppv.csv')
make_json(top_ppv_cot, list(range(1, 11)), ppv_store_path, 'ppv')

# make top sensitivity json files
top_sensitivity_cot = pd.read_csv(path + 'top_patterns_tpr.csv')
make_json(top_sensitivity_cot, list(range(1, 11)), sensitivity_store_path, 'sensitivity')

# make bottom ppv json files
bottom_ppv_cot = pd.read_csv(path + 'bottom_patterns_ppv.csv')
make_json(bottom_ppv_cot, list(range(11, 21)), ppv_store_path, 'ppv')

# make bottom sensitivity json files
bottom_sensitivity_cot = pd.read_csv(path + 'bottom_patterns_tpr.csv')
make_json(bottom_sensitivity_cot, list(range(11, 21)), sensitivity_store_path, 'sensitivity')

