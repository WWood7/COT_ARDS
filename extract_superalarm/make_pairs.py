import numpy as np
import pandas as pd
import ast
import json

superalarm_path = '/Users/winnwu/projects/Hu_lab/COT_project/generate/superalarm/'
matrix_path = '/Users/winnwu/projects/Hu_lab/COT_project/generate/tokens/matrix/'
pattern_number = 10
ppv_store_path = '/Users/winnwu/projects/Hu_lab/COT_project/generate/extracted_alarms/tokens_ppv/'
sensitivity_store_path = '/Users/winnwu/projects/Hu_lab/COT_project/generate/extracted_alarms/tokens_sensitivity/'
fake_store_path = '/Users/winnwu/projects/Hu_lab/COT_project/generate/extracted_alarms/tokens_fake/'
pairs_store_path = '/Users/winnwu/projects/Hu_lab/COT_project/generate/extracted_alarms/pairs/'


# make ppv pairs
for i in range(1, 11):
    pairs = {
        'id': i,
        'patterns': [
        '/cot/tokens_ppv/' + str(i) + '.json',
        '/cot/tokens_ppv/' + str(i + 10) + '.json',
        ],
        'reference': '/cot/referenceRanges.json',
        'type': 'ppv_pair'
    }
    with open(pairs_store_path + str(i) + '.json', 'w', encoding='utf-8') as json_file:
        json.dump(pairs, json_file, ensure_ascii=False, indent=4)
# make sensitivity pairs
for i in range(11, 21):
    pairs = {
        'id': i,
        'patterns': [
        '/cot/tokens_sensitivity/' + str(i - 10) + '.json',
        '/cot/tokens_sensitivity/' + str(i) + '.json',
        ],
        'reference': '/cot/referenceRanges.json',
        'type': 'sensitivity_pair'
    }
    with open(pairs_store_path + str(i) + '.json', 'w', encoding='utf-8') as json_file:
        json.dump(pairs, json_file, ensure_ascii=False, indent=4)

