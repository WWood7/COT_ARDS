import numpy as np
import matplotlib.pyplot as plt
import ast

generate_path = '/Users/winnwu/projects/Hu_lab/COT_project/generate/'
# read in the hit array
case_test = np.load(generate_path + 'tokenarray/case_test_HitArray_dict_0.1_sparse.npy', allow_pickle=True).item()
case_test_keys = list(case_test.keys())
# control_test = np.load(generate_path + 'tokenarray/control_test_HitArray_dict_0.05_sparse.npy', allow_pickle=True).item()
# control_test_keys = list(control_test.keys())

# for case test patients, index = 9, 36, 38, 44, 64 have good raster plots
index = 9
case_test_hit = case_test[case_test_keys[index]]['sparseHitArray']
case_test_time = case_test[case_test_keys[index]]['HitT']
case_test_dense_hit = case_test_hit.toarray()

case_test_time = case_test_time - np.max(case_test_time)


# get the positions for eventplots
case_test_pos = []
for i in range(case_test_dense_hit.shape[0]):
    case_test_subpos = []
    for j in range(case_test_dense_hit.shape[1]):
        if case_test_hit[i, j] == 1:
            case_test_subpos.append(case_test_time[j])
            print(case_test_subpos)
    case_test_pos.append(np.array(case_test_subpos))
# plt.eventplot(case_test_pos, color='black', linelengths=2)
# plt.xlabel('Relative Time to Onset (hours)')
# plt.ylabel('COT sets')
# plt.show()


x_values = []
y_values = []
# Loop through each category/set
for i, events in enumerate(case_test_pos):
    for event in events:
        x_values.append(event)  # Append the time of each event to x_values
        y_values.append(i)  # Append the category index to y_values

# Now plot these as dots
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
plt.scatter(x_values, y_values, alpha=0.6, s=10)  # Play with the alpha value for transparency

# Customizing the plot
plt.xlabel('Relative Time to ARDS Onset(hours)', fontsize=12)
plt.ylabel('Indices of COT sets', fontsize=12)
plt.xticks(range(0, -25, -6))
plt.xlim([-24.3, 0.3])
plt.title('Trigger-Based COT Vectors', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, axis='x')  # Add a grid for better readability

# Adjust axes limits as necessary
# plt.xlim([min_time, max_time])  # Uncomment and adjust as per your data
# plt.ylim([min_category, max_category])  # Uncomment and adjust as per your data

# Show the plot
plt.show()


# for i in range(70, 100, 1):
#     control_test_hit = control_test[control_test_keys[i]]['sparseHitArray']
#     control_test_dense_hit = control_test_hit.toarray()
#     print(np.shape(control_test_dense_hit))
#     if np.sum(control_test_dense_hit) > 0:
#         control_test_time = control_test[control_test_keys[i]]['HitT']
#         control_test_time = control_test_time - np.max(control_test_time)
#
#         control_test_pos = []
#         for k in range(control_test_hit.shape[0]):
#             control_test_subpos = []
#             for j in range(control_test_hit.shape[1]):
#                 if control_test_hit[k, j] == 1:
#                     control_test_subpos.append(control_test_time[j])
#             control_test_pos.append(np.array(control_test_subpos))
#         plt.eventplot(control_test_pos, color='black', linelengths=2)
#         plt.xlabel('Relative Time to End of Segment (hours)')
#         plt.ylabel('COT sets')
#         print(i)
#         plt.show()
#         break






