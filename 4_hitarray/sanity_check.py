import numpy as np
import pandas as pd

# tokenarray_path = '/Users/winnwu/projects/Hu_Lab/COT_project/generate/tokenarray/'
another_path = '/Users/winnwu/projects/Hu_Lab/COT_project/tokenarray/'
case_train_seg = pd.read_csv('/Users/winnwu/projects/Hu_Lab/COT_project/generate/segments/train_case_segs.csv')
print(len(case_train_seg))

# # check hitarray
# # data = np.load(tokenarray_path + 'case_test_Hitarray_dict_0.4_sparse.npy', allow_pickle=True).item()
# data = np.load(another_path + 'case_train_Hitarray_dict_0.4_sparse.npy', allow_pickle=True).item()
# print(data)
# print(len(data))

# # check tokenarray
# data = np.load(tokenarray_path + 'case_test_TokenArray_dict.npy', allow_pickle=True).item()
# outer_key = list(data.keys())
# print(outer_key[0])
# print(data[outer_key[0]])
#
# # check toolbox
# data = np.load(tokenarray_path + 'case_test_toolbox_input_0.4_sparse.npy', allow_pickle=True)
#
# print(data)

