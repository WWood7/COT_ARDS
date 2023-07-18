from offline_analysis_functions import *


generate_path = '/Users/winnwu/projects/emory-hu lab/COT_project/generate/mimiciv_lin/'
FPR_MAX_list = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

# preprocessing, generate code matrices
GenerateCodeMatrix4Whole(generate_path)

optimal_folder_path = generate_path.replace('mimiciv_lin/', '') + 'minsup/optimal/'
with open(optimal_folder_path + 'optimal_minsup_list.txt', 'r') as file:
    # read in the contents of the file as a string
    contents = file.read()
# use ast.literal_eval() to convert the string to a list
optimal_minsup_list = ast.literal_eval(contents)
GenerateOfflineResults(
    generate_path, FPR_MAX_list, optimal_minsup_list)