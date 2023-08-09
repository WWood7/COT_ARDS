import numpy as np
import glob
from scipy.sparse import load_npz
import ast

file = ('/Users/winnwu/projects/Hu_Lab/COT_project/generate/WAOR_files/'
        'case_trainData_allFea_GWAOR_0.15_sparse_comb_0.npz')

data = load_npz(file).todense()

print(data.shape)
print(type(data))

with open ('/Users/winnwu/projects/Hu_Lab/COT_project/generate/superalarm/superalarm_0.15.txt') as f:
    content = f.read()
    list_of_lists = ast.literal_eval(content)
print(len(list_of_lists))
print(len(np.unique(list(data[:, -1]))))

