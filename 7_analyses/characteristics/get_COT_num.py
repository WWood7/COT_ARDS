import ast
import glob

FPR_MAX_list = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
generate_path = '/Users/winnwu/projects/Hu_lab/COT_project/generate/'
superalarm_path = generate_path + 'superalarm/'

# Define the pattern to match files of interest
pattern = superalarm_path + 'superalarm_*.txt'
# Use glob to find files matching the pattern
files = glob.glob(pattern)
print(files)
# Loop through each matching file
for file_path in files:
    print(f"Processing {file_path}:")

    # Open and read the file
    with open(file_path, 'r') as file:
        for line in file:
            # Convert the line from a string representation of a list to an actual list
            python_list = ast.literal_eval(line)
            # Count the number of elements in the list
            count = len(python_list)

            print(count)