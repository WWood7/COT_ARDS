import numpy as np
from scipy.sparse import load_npz
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gamma
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import os
def weightingFuncGWAOR(hitArray, deltaT, a, b):
    return hitArray @ gamma.pdf(deltaT, a=a, scale=1 / b)

generate_path = '/Users/winnwu/projects/Hu_Lab/COT_project/generate/'
# set the store path
store_path = generate_path + 'XGB_results'
if not os.path.exists(store_path):
    os.makedirs(store_path)

train_path = generate_path + '/WAOR_files/'
# FPR_max = 0.3
# combination_idex = 0
# alpha = 1
# beta = 2

# get all the hyperparameter combinations
FPR_max = 0.15
centerTimeinMins_list = [10, 30, 50, 60, 90]
optFea_WAOR_list = {'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'b': [6, 3, 2, 1.5, 1.2, 1, 6. / 7, 0.75, 6. / 9, 0.6]}
combination = []
for centerTimeinMins in centerTimeinMins_list:
    for alpha in optFea_WAOR_list['a']:
        for beta in optFea_WAOR_list['b']:
            combination.append([centerTimeinMins, alpha, beta])
print(len(combination))

# using 5-fold cross validation to get the best hyperparameter combination
max_auc = 0
for combination_idex in range(len(combination)):
    print('combination_idex: ', combination_idex)
    train_case_data = (
            load_npz(train_path + 'case_trainData_allFea_GWAOR_' + str(FPR_max) +
                     '_sparse_comb_' + str(combination_idex) + '.npz').
            toarray())
    train_control_data = (
            load_npz(train_path + 'control_trainData_allFea_GWAOR_' + str(FPR_max) +
                     '_sparse_comb_' + str(combination_idex) + '.npz').
            toarray())
    train_data = np.concatenate((train_case_data, train_control_data))
    # -2: labels, -1: ids
    train_X = train_data[:, :-2]
    scaler = MinMaxScaler()
    normalized_train_X = scaler.fit_transform(train_X)
    train_y = train_data[:, -2]
    # conduct 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    temp_auc = []
    for train_index, test_index in kf.split(normalized_train_X):
        X_train, X_val = normalized_train_X[train_index], normalized_train_X[test_index]
        y_train, y_val = train_y[train_index], train_y[test_index]
        clf = xgb.XGBClassifier(objective='binary:logistic')
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_val)[:, 1]
        temp_auc.append(roc_auc_score(y_val, y_pred))
    # get the mean auc of 5-fold cross validation
    auc = np.mean(temp_auc)
    if auc > max_auc:
        max_auc = auc
        best_combination = {'index': combination_idex, 'params': combination[combination_idex]}

# store the best hyperparameter combination
with open(store_path + '/best_combination' + str(FPR_max) + '.txt', 'w') as f:
    f.write('best_combination: ' + str(best_combination) + '\n')
    f.write('max_auc: ' + str(max_auc) + '\n')







