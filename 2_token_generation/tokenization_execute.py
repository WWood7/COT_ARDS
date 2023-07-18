import pandas as pd
import numpy as np
from sklearn import model_selection
from tokenization_function_refined import *

data_path = '/Users/winnwu/projects/emory-hu lab/COT_project/data/'
generate_path = '/Users/winnwu/projects/emory-hu lab/COT_project/generate/'

# #
# # get vital tokens
# # first preprocess the vital signs using resampling and forward imputation
# # save the imputed vital signs to a csv file
# initial_vitals = pd.read_csv(data_path + 'Allvitals_adults.csv')
# imputed_vitals = impute_vitals(initial_vitals)
# imputed_vitals.to_csv(data_path + 'imputed_vitals.csv')

# then calculate the features for each segment
# save the features to a csv file
imputed_vitals = pd.read_csv(data_path + 'imputed_vitals.csv')
train_case_segs = pd.read_csv(generate_path + 'segments/train_case_segs.csv')
test_case_segs = pd.read_csv(generate_path + 'segments/test_case_segs.csv')
train_control_segs = pd.read_csv(generate_path + 'segments/train_control_segs.csv')
test_control_segs = pd.read_csv(generate_path + 'segments/test_control_segs.csv')
# CalculateFeatures(train_case_segs, test_case_segs, train_control_segs, test_control_segs,
#                   imputed_vitals, generate_path)
# predictors = ['segment_num',
#               'slope_pos_max', 'slope_pos_min', 'slope_pos_median', 'slope_pos_mean',
#               'slope_neg_max', 'slope_neg_min', 'slope_neg_median', 'slope_neg_mean',
#               'slope_pos_percent', 'slope_pos_duration_percent',
#               'slope_neg_percent', 'slope_neg_duration_percent',
#               'pos_slope_max_min_ratio', 'neg_slope_max_min_ratio',
#               'slope_change_rate_gt10_num', 'slope_change_rate_gt20_num',
#               'slope_change_rate_gt30_num', 'slope_change_rate_gt40_num',
#               'slope_change_rate_gt50_num', 'slope_change_rate_gt60_num',
#               'slope_change_rate_gt70_num', 'slope_change_rate_gt80_num',
#               'slope_change_rate_gt90_num', 'slope_change_rate_gt100_num',
#               'terminal_max', 'terminal_min', 'terminal_median', 'terminal_mean',
#               'DTposdur1', 'DTnegdur1', 'DTterminal1', 'DTslope1',
#               'DTposdur2', 'DTnegdur2', 'DTterminal2', 'DTslope2',
#               'th_DTterminal_ratio', 'th_DTslope_lastup_ratio', 'th_DTslope_lastdown_ratio']
# formTPFP4EachFeature(generate_path, predictors)
# pickup_top40token4EachVital(generate_path, predictors)
# formattedVitalTokenInput(generate_path)



# # get lab tokens
# # mark abnormal high or low for lab events with abnormal flags
# Getabnomral_labs_wLH(data_path)

# # get tokens
# abnormal_labs_wLH = pd.read_csv(data_path + 'abnormal_labs_wLH.csv')
# train_case_lab_token, test_case_lab_token,\
# train_control_lab_token, test_control_lab_token = \
#     GetLabTokens(train_case_segs, test_case_segs, train_control_segs, test_control_segs, abnormal_labs_wLH)
# lab_tokens_folder = generate_path + 'tokens/lab/'
# if not os.path.exists(lab_tokens_folder):
#     os.makedirs(lab_tokens_folder)
# train_case_lab_token.to_csv(lab_tokens_folder + 'train_case_lab_token_input.csv')
# test_case_lab_token.to_csv(lab_tokens_folder + 'test_case_lab_token_input.csv')
# train_control_lab_token.to_csv(lab_tokens_folder + 'train_control_lab_token_input.csv')
# test_control_lab_token.to_csv(lab_tokens_folder + 'test_control_lab_token_input.csv')


#
# get vent tokens
vent_method = 'intub_duration'
abnormal_vent = pd.read_csv(data_path + 'abnormal_vent_m_s.csv')
abnormal_vent = abnormal_vent.dropna(subset=["tokenid"])
train_case_vent_token, test_case_vent_token,\
train_control_vent_token, test_control_vent_token = \
    GetVentTokens(train_case_segs, test_case_segs, train_control_segs, test_control_segs, abnormal_vent, vent_method)
vent_token_folder = generate_path + 'tokens/vent'
if not os.path.exists(vent_token_folder):
    os.makedirs(vent_token_folder)
train_case_vent_token.to_csv(vent_token_folder + '/train_case_vent_' + vent_method + '_token_input.csv')
test_case_vent_token.to_csv(vent_token_folder + '/test_case_vent_' + vent_method + '_token_input.csv')
train_control_vent_token.to_csv(vent_token_folder + '/train_control_vent_' + vent_method + '_token_input.csv')
test_control_vent_token.to_csv(vent_token_folder + '/test_control_vent_' + vent_method + '_token_input.csv')


vent_method = 'all'
train_case_vent_token, test_case_vent_token,\
train_control_vent_token, test_control_vent_token = \
    GetVentTokens(train_case_segs, test_case_segs, train_control_segs, test_control_segs, abnormal_vent, vent_method)
train_case_vent_token.to_csv(vent_token_folder + '/train_case_vent_' + vent_method + '_token_input.csv')
test_case_vent_token.to_csv(vent_token_folder + '/test_case_vent_' + vent_method + '_token_input.csv')
train_control_vent_token.to_csv(vent_token_folder + '/train_control_vent_' + vent_method + '_token_input.csv')
test_control_vent_token.to_csv(vent_token_folder + '/test_control_vent_' + vent_method + '_token_input.csv')


vent_method = 'lasttwo'
train_case_vent_token, test_case_vent_token,\
train_control_vent_token, test_control_vent_token = \
    GetVentTokens(train_case_segs, test_case_segs, train_control_segs, test_control_segs, abnormal_vent, vent_method)
train_case_vent_token.to_csv(vent_token_folder + '/train_case_vent_' + vent_method + '_token_input.csv')
test_case_vent_token.to_csv(vent_token_folder + '/test_case_vent_' + vent_method + '_token_input.csv')
train_control_vent_token.to_csv(vent_token_folder + '/train_control_vent_' + vent_method + '_token_input.csv')
test_control_vent_token.to_csv(vent_token_folder + '/test_control_vent_' + vent_method + '_token_input.csv')


#
# get demographic tokens
# (age(18-44,45-64,65+), gender(male, female), ethnicity(black, white, asian, hispanic, others),
# height(-150, 150-160,160-170,170-180,180-190,190+), BMI(-18.5，18.5-25，25-30，30-35，35+))
# demographics = pd.read_csv(data_path + 'mimiciii_demographics.csv')
# demo_tokens_folder = generate_path + 'tokens/demo'
# if not os.path.exists(demo_tokens_folder):
#     os.makedirs(demo_tokens_folder)
# train_case_demo_token_input, test_case_demo_token_input,\
# train_control_demo_token_input, test_control_demo_token_input = \
#     GetDemoTokens(train_case_segs, test_case_segs, train_control_segs, test_control_segs, demographics)
# train_case_demo_token_input.to_csv(demo_tokens_folder + '/train_case_demo_token_input.csv')
# test_case_demo_token_input.to_csv(demo_tokens_folder + '/test_case_demo_token_input.csv')
# train_control_demo_token_input.to_csv(demo_tokens_folder + '/train_control_demo_token_input.csv')
# test_control_demo_token_input.to_csv(demo_tokens_folder + '/test_control_demo_token_input.csv')


#
# generate maps
GenerateVitalMap(generate_path)
GenerateLabMap(data_path, generate_path)
GenerateDemoMap(generate_path)
GenerateVentMap(data_path, generate_path, 'intub_duration')
GenerateVentMap(data_path, generate_path, 'all')
GenerateVentMap(data_path, generate_path, 'lasttwo')



