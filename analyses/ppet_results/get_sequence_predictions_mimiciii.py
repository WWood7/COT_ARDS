import numpy as np
from scipy.sparse import load_npz
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gamma
import xgboost as xgb
import os
import ast
import itertools

FPR_max = 0.25
def weightingFuncGWAOR(hitArray, deltaT, a, b):
    return hitArray @ gamma.pdf(deltaT, a=a, scale=1 / b)

generate_path = '/Users/winnwu/projects/Hu_Lab/COT_project/generate/'
train_path = generate_path + 'WAOR_files/'
test_path = generate_path + 'tokenarray/'
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
print('alpha: ', alpha)
print('beta: ', beta)
index = combination['index']
print('index: ', index)

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
train_y = train_data[:, -2]

# train the model
clf = xgb.XGBClassifier(objective='binary:logistic', n_estimators=1000, seed=123, learning_rate=0.15)
clf.fit(train_X, train_y)


# read in the testing data
# case
test_case_hitarray = np.load(test_path + 'case_test_HitArray_dict_' + str(FPR_max) +
                                '_sparse.npy', allow_pickle=True).item()
test_case_id = list(test_case_hitarray.keys())
test_case_seq_toolbox_input = []
for i in test_case_id:
    hitarray = test_case_hitarray[i]['sparseHitArray'].todense()
    hittime = test_case_hitarray[i]['HitT']
    final_time = hittime[-1]
    for j in range(hitarray.shape[1]):
        vector = np.array(weightingFuncGWAOR(hitarray[:, :j], hittime[j] - hittime[:j], alpha, beta))
        prob = clf.predict_proba(vector.reshape(1, -1))[0][1]
        timestamp = final_time - hittime[j]
        test_case_seq_toolbox_input.append([i, timestamp, prob])
test_case_seq_toolbox_input = np.array(test_case_seq_toolbox_input)
np.save(
    store_path + '/test_case_seq_toolbox_input' + str(FPR_max) +'.npy', test_case_seq_toolbox_input)

# control
test_control_hitarray = np.load(test_path + 'control_test_HitArray_dict_' + str(FPR_max) +
                                '_sparse.npy', allow_pickle=True).item()
test_control_id = list(test_control_hitarray.keys())
test_control_seq_toolbox_input = []
for i in test_control_id:
    hitarray = test_control_hitarray[i]['sparseHitArray'].todense()
    hittime = test_control_hitarray[i]['HitT']
    final_time = hittime[-1]
    for j in range(hitarray.shape[1]):
        vector = np.array(weightingFuncGWAOR(hitarray[:, :j], hittime[j] - hittime[:j], alpha, beta))
        prob = clf.predict_proba(vector.reshape(1, -1))[0][1]
        timestamp = final_time - hittime[j]
        test_control_seq_toolbox_input.append([i, timestamp, prob])
test_control_seq_toolbox_input = np.array(test_control_seq_toolbox_input)
np.save(
    store_path + '/test_control_seq_toolbox_input' + str(FPR_max) +'.npy', test_control_seq_toolbox_input)

