import numpy as np
import matplotlib.pyplot as plt

maxFPR = 0.1
generate_path = '/Users/winnwu/projects/Hu_lab/COT_project/generate/'
figure_path = generate_path + 'figures/'

# case
# read in the hit array
case_test = np.load(generate_path + 'tokenarray/case_test_HitArray_dict_' + str(maxFPR)
                    + '_sparse.npy', allow_pickle=True).item()
case_test_keys = list(case_test.keys())
num_of_COTs = np.shape(case_test[case_test_keys[1]]['sparseHitArray'].toarray())[0]

# generate a binned event count array
case_event_count = np.zeros((num_of_COTs, 24))
for i in range(len(case_test_keys)):
    case_test_hit = case_test[case_test_keys[i]]['sparseHitArray'].toarray()
    case_test_time = np.array(case_test[case_test_keys[i]]['HitT'])
    case_test_time = case_test_time - np.max(case_test_time)
    for j in range(24):
        # get the index of the events that fall into the j-th hour
        column_index = np.where((case_test_time > -(j + 1)) & (case_test_time <= -j))
        if len(column_index) != 0:
            select_columns = case_test_hit[:, column_index[0]]
            case_event_count[:, 23 - j] += np.sum(select_columns, axis=1)

# control
control_test = np.load(generate_path + 'tokenarray/control_test_HitArray_dict_' + str(maxFPR)
                          + '_sparse.npy', allow_pickle=True).item()
control_test_keys = list(control_test.keys())

# generate a binned event count array
control_event_count = np.zeros((num_of_COTs, 24))
for i in range(len(control_test_keys)):
    control_test_hit = control_test[control_test_keys[i]]['sparseHitArray'].toarray()
    control_test_time = np.array(control_test[control_test_keys[i]]['HitT'])
    control_test_time = control_test_time - np.max(control_test_time)
    for j in range(24):
        # get the index of the events that fall into the j-th hour
        column_index = np.where((control_test_time > -(j + 1)) & (control_test_time <= -j))
        if len(column_index) != 0:
            select_columns = control_test_hit[:, column_index[0]]
            control_event_count[:, 23 - j] += np.sum(select_columns, axis=1)
print(control_event_count)
vmin = min(case_event_count.min(), control_event_count.min())
vmax = max(control_event_count.max(), case_event_count.max())


# Plot the heatmap
plt.figure(figsize=(10, 6))
heatmap = plt.imshow(case_event_count, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
plt.colorbar(heatmap)

# Set the title and labels
plt.title('Trigger-Based COT Heatmap')
plt.xlabel('Hours Before ARDS Onset')
plt.ylabel('Indices of COT Sets')

# Assuming the x-axis currently goes from 0 to 23 (for 24 hours)
# We'll change it to go from -24 to -1
num_hours = case_event_count.shape[1]
hours_labels = np.arange(-24, 0, 1)
print(hours_labels)
plt.xticks(np.arange(num_hours), hours_labels)

plt.savefig(figure_path + 'case_heatmap_' + str(maxFPR) + '.png')



# Plot the heatmap
plt.figure(figsize=(10, 6))
heatmap = plt.imshow(control_event_count, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
plt.colorbar(heatmap)

# Set the title and labels
plt.title('Trigger-Based COT Heatmap')
plt.xlabel('Hours Before End of Sampled Segment')
plt.ylabel('Indices of COT Sets')

# Assuming the x-axis currently goes from 0 to 23 (for 24 hours)
# We'll change it to go from -24 to -1
num_hours = control_event_count.shape[1]
hours_labels = np.arange(-24, 0, 1)
print(hours_labels)
plt.xticks(np.arange(num_hours), hours_labels)
plt.savefig(figure_path + 'control_heatmap_' + str(maxFPR) + '.png')
