"""
This module contains three classes: augment, score, and process
And one function: "run" which performs scoring for the data and cleans up the output for easier processing

The highest level class is Process, which dictates how the data is processed and how
warnings(On/ Off) are defined. It takes the data and augments it according to the augmenter function
and computes scores according to the scoring functions that are passed. The rationale for this structure is
that we want to be resourceful and compute all scores of interest at the innermost level of the data
processing class. In this way we do not process the data independently for each scoring function.

"""
from typing import Callable, Sequence

import numpy as np


class Process(object):
    """
    This class contains high level data processing methods that are passed functions
    needed to augment data and calculate metrics/scores of interest

    """
    def __init__(self, scorers: Sequence[Callable], augmenter: Callable, thresholds: Sequence[float], **kwargs):
        """
        Must be initialized with an iterable of callable functions that each take in
        data as an argument (scorers), a function to augment the data (augmenter),
        as well as an iterable of thresholds to apply to the data

        :param scorers: iterable containing multiple scoring functions
        :param augmenter: data augmenting function
        :param thresholds: iterable containing thresholds to apply to the data
        :return:
        """
        self.scorers = scorers
        self.augmenter = augmenter

        self.thresh = thresholds

        for attribute, value in kwargs.items():
            setattr(self, attribute, value)

    def per_data(self, data: np.ndarray):
        """
        This function iterates through data at the encounter level and applies the combinations of
        scorers and augmenter (for each threshold), that are passed in the init.

        As the name suggests, warnings are created by thresholding the supplied model outputs at each data/time point.
        No special definitions are used to define how long model outputs must stay above the threshold in order
        to be counted as warnings (On). This is the standard thresholding approach.

        :param data: array containing model outputs for all encounters. (ID, Time (h), Model Output). No assumptions are
        made for the model output, which may be any real valued number (-inf, +inf).
        :return:
        """
        scores = []
        # Iterate through unique encounter ID's
        for patient in np.unique(data[:, 0]):
            # Select all data for the current encounter
            cur_data = data[data[:, 0] == patient, :]

            # Augment the data
            augmented_data = self.augmenter(cur_data)

            # Iterate through the augmented versions of the data
            cur_scores = []
            for cur_aug_data in augmented_data:
                cur_aug_scores = []
                # Iterate through the scoring functions, calculate scores, and store them
                for scorer in self.scorers:
                    # For the current scorer, calculate the score for each threshold supplied
                    temp = []
                    for thresh in self.thresh:
                        temp.append(scorer(data=cur_aug_data, threshold=thresh))
                    cur_aug_scores.append(temp)
                cur_scores.append(cur_aug_scores)
            # Store all scores calculated for the current encounter. Note we use lists since the outputs of
            # each scoring function are not guaranteed to have the same dimensions
            scores.append([[x[index] for x in cur_scores] for index in range(len(self.scorers))])
        return scores

    def stayon(self, data: np.ndarray):
        """
        CURRENTLY BROKEN

        This function iterates through data at the encounter level and applies the combinations of
        scorers and augmenter (for each threshold), that are passed in the init.

        As the name suggests, warnings are created by thresholding the supplied model outputs and defining a warning as On
        once a threshold is crossed and stays On unless the model output stays below the threshold for a certain time.

        :param data: array containing model outputs for all encounters. (ID, Time (h), Output). No assumptions are
        made for the model output, which may be probabilities [0,1] or any real valued number (-inf, +inf).
        :return:
        """
        assert all(hasattr(self, x) for x in ['stayon_time'])
        scores = []
        for patient in np.unique(data[:, 0]):
            cur_data = data[data[:, 0] == patient, :]

            augmented_data = self.augmenter(cur_data)

            cur_scores = []
            for cur_aug_data in augmented_data:
                cur_aug_scores = []
                for scorer in self.scorers:
                    temp_score = []
                    sorted_inds = np.argsort(cur_aug_data[:, 1])[::-1]
                    for thresh in self.thresh:
                        cur_aug_data_alt = cur_aug_data.copy()
                        split_inds = np.where(np.diff(np.where(cur_aug_data[sorted_inds, 2] < thresh)[0]) > 1)[0]
                        indexes = (0,) + tuple(dat + 1 for dat in split_inds) + (len(sorted_inds),)

                        pos_inds = sorted_inds
                        neg_inds = []
                        if len(indexes) > 2:
                            off_inds = []
                            l = [np.arange(start, end) for start, end in zip(indexes, indexes[1:]) if ((end - start) * self.interval) >= self.stayon_time]
                            if len(l) > 0:
                                off_inds = np.concatenate(l)

                            neg_inds = sorted_inds[off_inds]
                            pos_inds = np.delete(sorted_inds, sorted_inds[off_inds])

                        cur_aug_data_alt[pos_inds, 2] = thresh
                        cur_aug_data_alt[neg_inds, 2] = 0

                        temp_score.append(scorer(data=cur_aug_data_alt, threshold=thresh))
                    cur_aug_scores.append(temp_score)
                cur_scores.append(cur_aug_scores)

            scores.append([[x[index] for x in cur_scores] for index in range(len(self.scorers))])
        return scores

def run(data: np.ndarray, processor: Callable) -> "np.array":
    """
    This function takes a previously initialized instance of processor and the data to analyze

    :param data: Numpy array containing data for all patients Columns: (ID, Time (+ before event), Model Output)
    :param processor: Fully initialized Processor function
    :return:
        - counts - Array of scores summed up across patients for each random trial
        - raw_counts - Raw array of scores for all patients and all randomizations
    """

    # Ensure no data after the time of event is analyzed
    score = processor(data[data[:, 1] >= 0, :])
    # Calculate number of scorers
    n = len(score[0])
    # Store the results of all the scores as a list of arrays (# Scorers x # Patients x # Thresholds x Dimension of Scorer)
    raw_counts = [np.array([cur_score[index] for cur_score in score]) for index in range(n)]
    # Add up all the scores for each randomized trial (# Scorers x # Randomizations x # Thresholds x Dimension of Scorer)
    counts = [np.nansum(np.array([cur_score[index] for cur_score in score]), axis=0) for index in range(n)]

    return counts, raw_counts