from COT_mining_functions import *
import ast

generate_path = '/Users/winnwu/projects/emory-hu lab/COT_project/generate/'
MinSup_list = [0.1, 0.15, 0.2]
FPR_MAX_list = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
fold_num = 5

# # preprocessing, generate code matrices
# GenerateCodeMatrix4CV(generate_path)
# GenerateCodeMatrix4Whole(generate_path)
# GenerateOptimalMS(generate_path, MinSup_list, FPR_MAX_list, fold_num)

optimal_folder_path = generate_path + 'minsup/optimal/'
with open(optimal_folder_path + 'optimal_minsup_list.txt', 'r') as file:
    # read in the contents of the file as a string
    contents = file.read()
# use ast.literal_eval() to convert the string to a list
optimal_minsup_list = ast.literal_eval(contents)
with open(optimal_folder_path + 'final_FP_thresh_list.txt', 'r') as file:
    # read in the contents of the file as a string
    contents = file.read()
operating_point = ast.literal_eval(contents)

# GenerateOfflineResults_alreadygotoutputversion(
#     generate_path, optimal_minsup_list, operating_point, FPR_MAX_list)

GenerateOfflineResults_alreadyhavepatternsetversion(
    generate_path, FPR_MAX_list, optimal_minsup_list)
