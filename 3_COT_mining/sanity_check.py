import numpy as np
import ast
# check if the final counts of superalarm patterns align well with minimal support and FPR_MAX
superalarm_path = '/Users/winnwu/projects/emory-hu lab/COT_project/generate/superalarm/'
FPR_MAX_list = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
count = []
for i in FPR_MAX_list:
    file_path = superalarm_path + 'superalarm_' + str(i) + '.txt'
    with open(file_path, "r") as file:
        contents = file.read()
    superalarm_list = ast.literal_eval(contents)

    count.append(len(superalarm_list))

print(count)