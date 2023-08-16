import pandas as pd
data_path = '/Users/winnwu/projects/Hu_Lab/COT_project/data/mimiciv/'
myaddedlabventduration = pd.read_csv(data_path + 'mimiciv_ventilation.csv')
allvents = pd.read_csv(data_path + 'abnormal_vent_m_s.csv')
print(myaddedlabventduration.loc[myaddedlabventduration['stay_id'] == 30283714])