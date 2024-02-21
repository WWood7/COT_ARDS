import numpy as np

np.random.seed(1)

class ShuffleEvent(object):
    """
    This class shuffles the time of event (marked at t=0) by re-referencing
    all time stamps to a random time before the time of event. All data after the
    new time of event is then removed
    """
    def __init__(self, num_samples: int):
        """

        :param num_samples: Number of shuffled event time copies to generate
        """
        self.num_samples = num_samples

    def __call__(self, data):
        """
        :param data: Matrix of current patient data Columns: (ID, Time, Model Output)
        :return: List of data with randomly shifted event times
        """
        # Select num_samples indices
        inds = np.random.choice(len(data), self.num_samples)
        shifted_data = []
        for ind in inds:
            # Make a copy of the data to be safe
            cur_random_data = data.copy()
            # Re-reference the time
            cur_random_data[:, 1] -= data[ind, 1]
            # Store the new data, only including data *before* the new event time
            shifted_data.append(cur_random_data[cur_random_data[:, 1] >= 0, :])
        return shifted_data

class ShuffleEventMax(object):
    """
    This method shuffles the time of event (marked at t=0) by re-referencing
    all time stamps to a random time, no more than max_time before the time of event.
    """
    def __init__(self, num_samples: int, max_time: float):
        """

        :param max_time: Maximum time to shift the event time by
        :param num_samples: Number of shuffled event time copies to generate
        """
        self.num_samples = num_samples
        self.max_time = max_time

    def __call__(self, data: np.ndarray):
        """

        :param data: Matrix of current patient data Columns: (ID, Time, Score)
        :return: List of data with shuffled event times
        """
        # Check for candidate time points
        candidates = np.where(np.logical_and(data[:, 1] <= self.max_time, data[:, 1] >= 0))[0]
        # Condition if no candidate time points exist
        if len(candidates) == 0:
            # Make a copy of the data to be safe
            null_data = data.copy()
            # Make all time values <0 so they aren't counted in any downstream score calculation
            null_data[:, 1] = -1
            # Store num_sample # of copies
            return [null_data] * self.num_samples
        else:
            # Select num_samples # of candidate time points
            inds = np.random.choice(candidates, self.num_samples)
        shifted_data = []
        for ind in inds:
            # Make a copy of the data to be safe
            cur_data = data.copy()
            # Re-reference the time to the new time of event
            cur_data[:, 1] -= data[ind, 1]
            # Store the new data, only including data *before* the new event time
            shifted_data.append(cur_data[cur_data[:, 1] >= 0, :])
        return shifted_data

class NoAugment(object):
    def __call__(self, data: np.ndarray):
        """
        This method performs no augmentation and simply returns the data within a list

        :param data: Matrix of current patient data Columns: (ID, Time, Score)
        :return: List containing original data
        """
        return [data]

class RandomWindows(object):
    def __init__(self, num_samples: int, duration: int, is_sort: int):
        """
        Initializes the class with the number of windows to sample from the data

        :param num_samples: Number of random n hour windows to select
        :duration: the duration of window (n hours) to be selected
        :is_sort: whether to sort random windows by timeline
        """
        self.num_samples = num_samples
        self.duration = duration
        self.is_sort = is_sort

    def __call__(self, data: np.ndarray):
        """
        This method randomly samples self.duration hour windows from the data and re-references the time so that t=0
        gives the time that was closest to the time of event or discharge, for case and control data, respectively

        :param data: Matrix of current patient data Columns: (ID, Time, Score)
        :return:
            - replicated_data - List of randomly sampled data
        """
        replicated_data = []
        unique = 0
        # Find min and max times for current data
        times = data[:, 1]
        t0 = np.max(times)
        t1 = np.min(times)
        if abs(t0 - t1) >= self.duration:
            # Select candidate boundary times for our self.duration hour windows,
            # those that are at least self.duration hours before the time of discharge
            boundary = np.where(data[:, 1] >= self.duration)[0]
            # Check that there exist any valid candidate data points
            if len(boundary) > 0:
                # Select num_samples # of starting points for each window
                inds = np.random.choice(boundary, self.num_samples)
                if self.is_sort ==1:
                    inds = np.sort(inds)
                for ind in inds:
                    # Find the indexes for the current self.duration hour window selection
                    window = np.where(np.logical_and(times <= times[ind], times >= times[ind] - self.duration))[0]
                    cur_random_data = data[window, :].copy()
                    # Re-reference time as described above
                    cur_random_data[:, 1] -= np.min(cur_random_data[:, 1])
                    # Assert that all time points in the new data are >= 0
                    # This is a requirement for all scoring functions that will use this data
                    assert np.sum(cur_random_data[:, 1] >= 0) == cur_random_data.shape[0]
                    # Store the random self.duration hour window
                    replicated_data.append(cur_random_data)
            return replicated_data
        else:
            return [[]] * self.num_samples
        
class RandomWindows_case(object): #ranxiao: for evaluating FPR outside of prediction horizon and lead time
    def __init__(self, num_samples: int,t_horizon: int,t_lead: int):
        """
        Initializes the class with the number of windows to sample from the data

        :param num_samples: Number of random 12 hour windows to select
        """
        self.num_samples = num_samples
        self.t_horizon = t_horizon
        self.t_lead = t_lead
      

    def __call__(self, data: np.ndarray):
        """
        This method randomly samples 12 hour windows from the data and re-references the time so that t=0
        gives the time that was closest to the time of event or discharge, for case and control data, respectively

        :param data: Matrix of current patient data Columns: (ID, Time, Score)
        :return:
            - replicated_data - List of randomly sampled data
        """
        replicated_data = []
        unique = 0
        # Find min and max times for current data
        times = data[:, 1]
        t0 = np.max(times)
        t1 = np.min(times)
        if abs(t0 - t1) >= self.t_horizon*2+self.t_lead: #ranxiao: 12 hour for prediction horizon, another at least 12 hours for random 12-hour window selection
            # Select candidate boundary times for our 12 hour windows,
            # those that are at least 12 hours before the time of discharge
            boundary = np.where(data[:, 1] >= self.t_horizon*2+self.t_lead)[0]
            # Check that there exist any valid candidate data points
            if len(boundary) > 0:
                # Select num_samples # of starting points for each window
                inds = np.random.choice(boundary, self.num_samples)
                for ind in inds:
                    # Find the indexes for the current 12 hour window selection
                    window = np.where(np.logical_and(times <= times[ind], times >= times[ind] - 12))[0]
                    cur_random_data = data[window, :].copy()
                    # Re-reference time as described above
                    cur_random_data[:, 1] -= np.min(cur_random_data[:, 1])
                    # Assert that all time points in the new data are >= 0
                    # This is a requirement for all scoring functions that will use this data
                    assert np.sum(cur_random_data[:, 1] >= 0) == cur_random_data.shape[0]
                    # Store the random 12 hour window
                    replicated_data.append(cur_random_data)
            return replicated_data
        else:
            return [[]] * self.num_samples