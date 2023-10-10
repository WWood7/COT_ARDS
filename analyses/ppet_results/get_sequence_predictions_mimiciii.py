import numpy as np
from scipy.sparse import load_npz
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gamma
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import os
import ast

FPR_max = 0.15
def weightingFuncGWAOR(hitArray, deltaT, a, b):
    return hitArray @ gamma.pdf(deltaT, a=a, scale=1 / b)

generate_path = '/Users/winnwu/projects/Hu_Lab/COT_project/generate/'
train_path = generate_path + '/WAOR_files/'
test_path = generate_path + '/tokenarray/'
# set the store path
store_path = generate_path + 'XGB_results/mimiciii'
if not os.path.exists(store_path):
    os.makedirs(store_path)

# train a model using the best hyperparameter combination
# read in the best hyperparameter combination
optimal_params = generate_path + 'XGB_results/best_combination' + str(FPR_max) + '.txt'
with open(optimal_params, 'r') as f:
    lines = f.readlines()
dict_string = lines[0].split("best_combination: ")[1].strip()
combination = ast.literal_eval(dict_string)
alpha = combination['params'][1]
beta = combination['params'][2]
index = combination['index']

# read in the training data

train_case_data = (
        load_npz(train_path + 'case_trainData_allFea_GWAOR_' + str(FPR_max) +
                 '_sparse_comb_' + str(index) + '.npz').
        toarray())
train_control_data = (
        load_npz(train_path + 'control_trainData_allFea_GWAOR_' + str(FPR_max) +
                 '_sparse_comb_' + str(index) + '.npz').
        toarray())
train_data = np.concatenate((train_case_data, train_control_data))

# -2: labels, -1: ids
train_X = train_data[:, :-2]
scaler = MinMaxScaler()
train_y = train_data[:, -2]

# train the model
clf = xgb.XGBClassifier(objective='binary:logistic')
clf.fit(train_X, train_y)


# read in the testing data
test_case_hitarray = np.load(test_path + 'tokenarray/case_HitArray_dict_' + str(FPR_max) +
                                '_sparse.npy', allow_pickle=True).item()
test_case_id = list(test_case_hitarray.keys())