from sklearn.preprocessing import normalize
import numpy as np
from scipy.sparse import load_npz
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import gamma
def weightingFuncGWAOR(hitArray, deltaT, a, b):
    return hitArray @ gamma.pdf(deltaT, a=a, scale=1 / b)

generate_path = '/Users/winnwu/projects/Hu_Lab/COT_project/generate/'
FPR_max = 0.3
combination_idex = 0
alpha = 1
beta = 2

train_case_data = (
        load_npz(generate_path + 'WAOR_files/case_trainData_allFea_GWAOR_' + str(FPR_max) +
                 '_sparse_comb_' + str(combination_idex) + '.npz').
        toarray())
train_control_data = (
        load_npz(generate_path + 'WAOR_files/control_trainData_allFea_GWAOR_' + str(FPR_max) +
                 '_sparse_comb_' + str(combination_idex) + '.npz').
        toarray())
train_data = np.concatenate((train_case_data, train_control_data))
train_X = train_data[:, :-2]
scaler = MinMaxScaler()
normalized_train_X = scaler.fit_transform(train_X)
train_y = train_data[:, -2]

rf_clf = RandomForestClassifier().fit(X=train_X, y=train_y)
reg_clf = LogisticRegression(max_iter=1000).fit(X=train_X, y=train_y)

# test_case_data = (
#         load_npz(generate_path + 'WAOR_files/case_testData_allFea_GWAOR_' + str(FPR_max) +
#                  '_sparse_comb_' + str(combination_idex) + '.npz').
#         toarray())
# test_control_data = (
#         load_npz(generate_path + 'WAOR_files/control_testData_allFea_GWAOR_' + str(FPR_max) +
#                  '_sparse_comb_' + str(combination_idex) + '.npz').
#         toarray())
# test_data = np.concatenate((test_case_data, test_control_data))
# test_X = test_data[:, :-2]
# test_y = test_data[:, -2]
# print(rf_clf.score(test_X, test_y))

test_case_hitarray = np.load(generate_path + 'tokenarray/case_test_HitArray_dict_' + str(FPR_max) +
                                '_sparse.npy', allow_pickle=True).item()
test_case_id = list(test_case_hitarray.keys())
test_case_count_rf = 0
test_case_count_reg = 0
for i in test_case_id:
    rf_flag = 0
    reg_flag = 0
    hitarray = test_case_hitarray[i]['sparseHitArray'].todense()
    hittime = test_case_hitarray[i]['HitT']
    for j in range(hitarray.shape[1]):
        # if sum(hitarray[:, j]) > 0:
        vector = np.array(weightingFuncGWAOR(hitarray[:, :j], hittime[j] - hittime[:j], alpha, beta))
        y_rf = rf_clf.predict(vector.reshape(1, -1))
        y_reg = reg_clf.predict(vector.reshape(1, -1))
        if y_rf == 1:
            rf_flag = 1
        if y_reg == 1:
            reg_flag = 1
        if rf_flag == 1 and reg_flag == 1:
            break

    test_case_count_rf += rf_flag
    test_case_count_reg += reg_flag
print('TPR_rf:', test_case_count_rf / len(test_case_id))
print('TPR_reg:', test_case_count_reg / len(test_case_id))


test_control_hitarray = np.load(generate_path + 'tokenarray/control_test_HitArray_dict_' + str(FPR_max) +
                                '_sparse.npy', allow_pickle=True).item()
test_control_id = list(test_control_hitarray.keys())
test_control_count_reg = 0
test_control_count_rf = 0
for i in test_control_id:
    rf_flag = 0
    reg_flag = 0
    hitarray = test_control_hitarray[i]['sparseHitArray'].todense()
    hittime = test_control_hitarray[i]['HitT']
    for j in range(hitarray.shape[1]):
        # if sum(hitarray[:, j]) > 0:
        vector = np.array(weightingFuncGWAOR(hitarray[:, :j + 1], hittime[j] - hittime[:j + 1], alpha, beta))
        y_rf = rf_clf.predict(vector.reshape(1, -1))
        if y_rf == 1:
            rf_flag = 1
        if y_reg == 1:
            reg_flag = 1
        if rf_flag == 1 and reg_flag == 1:
            break
    test_control_count_rf += rf_flag
    test_control_count_reg += reg_flag
print('FPR_rf:', test_control_count_rf / len(test_control_id))
print('FPR_reg:', test_control_count_reg / len(test_control_id))
