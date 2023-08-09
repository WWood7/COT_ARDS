import numpy as np
from scipy.sparse import csr_matrix, vstack, save_npz, load_npz
from scipy.stats import gamma
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GroupKFold

generate_path = '/Users/winnwu/projects/Hu_Lab/COT_project/generate/'
combination_idex = 0
FPR_max = 0.15
train_case_data = (
            load_npz(generate_path + 'WAOR_files/case_trainData_allFea_GWAOR_' + str(FPR_max) +
                     '_sparse_comb_' + str(combination_idex) + '.npz').
            toarray())
train_control_data = (
            load_npz(generate_path + 'WAOR_files/control_trainData_allFea_GWAOR_' + str(FPR_max) +
                     '_sparse_comb_' + str(combination_idex) + '.npz').
            toarray())
train_data = np.concatenate((train_case_data, train_control_data))
print(train_data.shape, sum(train_data[:, -2] == 1), sum(train_data[:, -2] == 0))
train_X = train_data[:, :-2]
normalized_train_X = normalize(train_X, axis=0)
train_y = train_data[:, -2]
print(normalized_train_X)
print(train_y)
indices_1 = np.where(train_data[:, -2] == 1)[0]
indices_0 = np.where(train_data[:, -2] == 0)[0]
group_1 = train_data[train_data[:, -2] == 1][:, -1]
group_0 = train_data[train_data[:, -2] == 0][:, -1]
gkf = GroupKFold(n_splits=5)
splits_1 = gkf.split(indices_1, groups=group_1)
splits_0 = gkf.split(indices_0, groups=group_0)
combined_splits = []
for (train_idx_0, val_idx_0), (train_idx_1, val_idx_1) in zip(splits_0, splits_1):
    train_idx = np.concatenate([indices_0[train_idx_0], indices_1[train_idx_1]])
    val_idx = np.concatenate([indices_0[val_idx_0], indices_1[val_idx_1]])
    print(sum(train_data[train_idx, -2] == 1), sum(train_data[train_idx, -2] == 0))
    print(sum(train_data[val_idx, -2] == 1), sum(train_data[val_idx, -2] == 0))