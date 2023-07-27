import numpy as np
import glob
from scipy.sparse import load_npz

# Get a list of all .npy files in the target directory with names starting with a certain string
files = \
    glob.glob('/Users/winnwu/projects/Hu_Lab/COT_project/generate/WAOR_files/'
              'case_trainData_allFea_GWAOR_0.05_sparse_comb_4patstart200143To299867.npz')

# Create an empty list to store individual numpy arrays
arrays_list = []

# Loop over the list of files
for file in files:
    # Load each .npy file into a numpy array and append it to the list
    arrays_list.append(load_npz(file).todense())

# Now, arrays_list contains all the numpy arrays loaded from the .npy files
print(arrays_list[0])
data = np.concatenate(arrays_list, axis=0)
print(data.shape)