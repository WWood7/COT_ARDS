import numpy as np

tokenarray_path = '/Users/winnwu/projects/Hu_Lab/COT_project/generate/tokenarray/'

# # check hitarray
# data = np.load(tokenarray_path + 'case_test_Hitarray_dict_0.4_sparse.npy', allow_pickle=True).item()
# data = data[200143]
# print(np.shape(data['sparseHitArray'].todense()))
# print(len(data['HitT']))


# check tokenarray
data = np.load(tokenarray_path + 'case_test_TokenArray_dict.npy', allow_pickle=True).item()
outer_key = list(data.keys())
print(outer_key[0])
print(data[outer_key[0]])

# check toolbox
data = np.load(tokenarray_path + 'case_test_toolbox_input_0.4_sparse.npy', allow_pickle=True)

print(data)

