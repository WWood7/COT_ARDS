import numpy as np

def DT_Sign(dur_slope, Sign):
    #This function finds the dominant trend start index and number of consecutive
    #windows for given slopes of overlapped windows and slope sign.
    #Input: dur_slope - slope of all the overlapped windows
    #       Sign - Either 1 or 0. 1 for '+' slope and 0 for '-' slope
    #Output: DT_start - start index of the dominant trend
    #        DT_L - number of consecutive windows

    dur_sign = np.sign(dur_slope).tolist()
    #print(dur_sign)

    DT_start=np.nan
    DT_L=0

    #Current window
    CW_start=np.nan
    CW_L=0

    #Those in dur_slope whose sign matches 'Sign' are converted to 1.
    #nan and not matching 'Sign' are converted to 0.
    #Then the problem becomes finding the longest 1's.
    Idx=[1 if val==Sign else 0 for val in dur_sign]
    #print(Idx)

    previous=0
    for i in range(len(Idx)):
        if Idx[i]==0: # if Idx(i) is 0
            previous = 0
        else: # if Idx(i) is 1
            if previous == 0: # start of 1's
                previous = 1
                CW_start = i
                CW_L = 1
            else: # in the middle of 1's
                previous = 1
                CW_L = CW_L + 1

            if CW_L > DT_L:
                DT_start = CW_start
                DT_L = CW_L

    return DT_start, DT_L