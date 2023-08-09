5This script is split into different modules, to run the whole project, run the files end with 'execute' in each folder, in numerical order.
The trend_processing folder contains files for l1 trend filtering and dominant trend detection, which are not directly used but will be utilized multiple times.

Through the processing of this project:
All related data files are stored in the data folder.
All sample pieces and intermediate files generated are stored in the generate folder.




Brief introduction of each module:


1. Preparatory work
This module is aimed at getting the targeted cohort of patients, split the data into test set and training set, as well as form segments for each sample. Additionally, maps of token ids together with corresponding token names will be generated.



2. token_generation
This module is aimed at generating required tokens for data from each source, including vital signs, ventilation events, demographic information and lab test results. After running this module, token inputs (within the prediction window before onset or end of segments) will be generated, which include token names, token ids and occurring time of the tokens (as time ahead of onset or end of segments, in hours).

A. Vital signs
Some trend features of 6 vital signs are considered as tokens. To solve temporal irregularity, median-based resampling and forward filling are used.
For l1 trend filtering, hyper parameters are selected by comparing TPR (choose the feature that has highest TPR but with FPR smaller than a threshold).
Note: the FPR threshold selected here is 0.05. If the threshold is too small that there does not exist a value of a feature that satisfies this FPR threshold, some problems shall occur. This can be checked by looking at the final top features table, features with value threshold of 0s indicate occurrence of this issue.

B. Lab test results
All numerical lab test results that do not lay in the normal range are counted as lab tokens.

C. Ventilation settings/measurements
Three kinds of tokens are included. The first one is intubation duration in hours (only look at the last intubation period). The second one is ventilation settings/measurements with abnormal values. The third one is the transition of ventilation settings or measurements, this kind of tokens are the last change of certain kinds of ventilation settings or measurements.

D. Demographic information
Demographic information includes height, weight, race, age, gender and BMI. These tokens are considered as static tokens and the generated input count all these static tokens as happen at onset or end of segments, which may be a little problematic.



3. COT_mining
This module is aimed at getting COT/SuperAlarm patterns using mafia algorithm, which can be carried out by a MafiaCPP.exe file built by Cheng Ding and Ran Xiao. For each FPR threshold (filter out the patterns with larger FPR) considered, identify the optimal (highest TPR) minimal support of Mafia by 5-fold cross validation. The results will be generated and stored under folder generate/minsup/optimal. Optimal_minsup_list.txt records all the optimal minimal supports for each FPR threshold, and the results of the final offline analysis (TPRs and FPRs) will be printed.



4. Hitarray
This module is for generating hit arrays for temporal related analysis. An hit array for one patient is a matrix indicating the trigger situations of the COT/SuperAlarm patterns, each row will be a certain pattern, and each column is a binary-valued pattern vector at a certain timestamp. 
After running this module, there will be 4 kinds of files generated in the /generate/tokenarray folder.
a. group_name_TokenArray_dict.npy
b. group_name_TokenTimeArray_dict.npy
c. group_name_Hitarray_dict_FPRmax_sparse.npy
d. group_name_toolbox_input_FPRmax_sparse.npy
See the following description of these files. All the relative durations are in hours.



group_name_TokenArray_dict.npy

A recommended way to load these files is use:
variable_name = np.load(file_name, allow_pickle=True).item()
In this way, the variable acquired will be a nested dictionaries. 
For the outer part, each key represents one patient’s icu_stay id, and the corresponding value is an inner dictionary that contains triggered tokens and timestamps. In an inner dictionary, each key is the duration from the first timestamp to the current timestamp, each value is a list of token ids indicating tokens that are in the state of triggered (note that not all the triggered tokens are necessarily triggered at this time point, some could be 	triggered earlier and then last).




group_name_TokenTimeArray_dict.npy

A recommended way to load these files is use:
variable_name = np.load(file_name, allow_pickle=True).item()
n this way, the variable acquired will be a nested dictionaries. 

For the outer part, each key represents one patient’s icu_stay id, and the corresponding value is an inner dictionary that contains tokens’ triggering time and timestamps. In an inner dictionary, each key is the duration from the first timestamp to the current timestamp, each value is a list of duration indicating the actual relative timestamp of a token being triggered. Together with group_name_TokenArray_dict.npy, one can see the exact timestamp of each token being triggered.




group_name_Hitarray_dict_FPRmax_sparse.npy

A recommended way to load these files is use:
variable_name = np.load(file_name, allow_pickle=True).item()
In this way, the variable acquired will be nested dictionaries. 

For the outer part, each key represents one patient’s icu_stay id, and the corresponding value is an inner dictionary that contains hit arrays and timestamps. In an inner dictionary, there are two keys ‘sparseHitArray’ and ‘HitT’. The value of ‘sparseHitArray’ is a sparse 0-1 matrix, each row is a COT or SuperAlarm pattern, each column represents a timestamp, and each element represents if a pattern is triggered at a timestamp. The value of ‘HitT’ is a list of relative timestamps, each element is the duration between the first stamp and the current stamp. The length of the timestamp list is equal to # of columns in the sparse matrix.




group_name_toolbox_input_FPRmax_sparse.npy

A recommended way to load these files is use:
variable_name = np.load(file_name, allow_pickle=True)
In this way, the variable acquired will be ndarrays with 3 columns. 

The first column contains icu_stay ids, the second contains timestamps, the third contains indicators representing if ANY COT or SuperAlarm patterns are triggered at that time point. Here the timestamps are relative duration from the current stamp to the last stamp, to be differentiate with timestamps in other files.



5. WAOR
This module is for using another way to represent the pattern vectors in hit arrays, termed as Weighted Average Occurrence Representation. Then use the new vectors to train classifiers.





For more detailed information, see the comments in code files.