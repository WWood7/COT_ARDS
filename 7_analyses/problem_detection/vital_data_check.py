import pandas as pd

mimiciii_vital = pd.read_csv('/Users/winnwu/projects/Hu_Lab/COT_project/data/AllVitals_adults.csv')
mimiciv_vital = pd.read_csv('/Users/winnwu/projects/Hu_Lab/COT_project/data/mimiciv/allvitals.csv')

mimiciii_id = mimiciii_vital['icustay_id'].unique()[3]
mimiciv_id = mimiciv_vital['stay_id'].unique()[1]

mimiciii_vital = mimiciii_vital[mimiciii_vital['icustay_id'] == mimiciii_id]
mimiciv_vital = mimiciv_vital[mimiciv_vital['stay_id'] == mimiciv_id]

print(mimiciii_vital['charttime'].sort_values(ascending=True).reset_index(drop=True))
print(mimiciv_vital['charttime'].sort_values(ascending=True).reset_index(drop=True))