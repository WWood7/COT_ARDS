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



4. WAOR_representation




For more detailed information, see the comments in code files.