import statistics
from typing import Sequence

import numpy as np


def worst_case(x: Sequence[float], vital_name: str):
    """
    Calculates worst case vital sign according to calculated MEWS scores

    :param x: Iterable of vital sign values
    :param vital_name: Name of the vital sign being passed
    :return: Worst case vital sign or NaN if they're all are NaN
    """
    indexes = [[], [], [], []]
    values = np.array([0, 1, 2, 3])
    x = np.array(x)
    if vital_name == 'sbp':
        indexes[3].append(np.where(x <= 70.)[0])
        indexes[2].append(np.where(np.logical_and(x >= 71., x <= 80.))[0])
        indexes[1].append(np.where(np.logical_and(x >= 81., x <= 100.))[0])
        indexes[0].append(np.where(np.logical_and(x >= 101., x <= 199.))[0])
        indexes[3].append(np.where(x >= 200.)[0])

    elif vital_name == 'hr':
        indexes[2].append(np.where(x <= 40.)[0])
        indexes[1].append(np.where(np.logical_and(x >= 41., x <= 50.))[0])
        indexes[0].append(np.where(np.logical_and(x >= 51., x <= 100.))[0])
        indexes[1].append(np.where(np.logical_and(x >= 101., x <= 110.))[0])
        indexes[2].append(np.where(np.logical_and(x >= 111., x <= 129.))[0])
        indexes[3].append(np.where(x >= 130.)[0])

    elif vital_name == 'resp':
        indexes[2].append(np.where(x < 9.)[0])
        indexes[0].append(np.where(np.logical_and(x >= 9., x <= 14.))[0])
        indexes[1].append(np.where(np.logical_and(x >= 15., x <= 20.))[0])
        indexes[2].append(np.where(np.logical_and(x >= 21., x <= 29.))[0])
        indexes[3].append(np.where(x >= 30)[0])

    elif vital_name == 'temp':
        indexes[2].append(np.where(x < 95.)[0])
        indexes[0].append(np.where(np.logical_and(x >= 95., x <= 101.12))[0])
        indexes[2].append(np.where(x >= 101.13)[0])

    elif vital_name == 'coma':
        indexes[0].append(np.where(np.logical_and(x >= 14., x <= 15.))[0])
        indexes[1].append(np.where(np.logical_and(x >= 10., x <= 13.))[0])
        indexes[2].append(np.where(np.logical_and(x >= 4., x <= 9.))[0])
        indexes[3].append(np.where(x == 3.)[0])

    indexes[:] = [np.concatenate(l) if len(l) > 0 else [] for l in indexes]

    inds = [len(l) != 0 for l in indexes]
    if any(inds):
        return x[indexes[int(np.max(values[inds]))][0]]
    else:
        return np.nan


def mode(x: Sequence):
    """
    Calculates the mode of an iterable
    :param x:
    :return:
    """
    vals, counts = np.unique(x, return_counts=True)
    if hasattr(vals, "__len__"):
        return vals[np.argmax(counts)]
    else:
        return vals


def find_max_mode(x: list):
    """
    Calculates mode and breaks ties by choosing the max

    :param x:
    :return:
    """
    list_table = statistics._counts(x)
    len_table = len(list_table)

    if len_table == 1:
        max_mode = statistics.mode(x)
    else:
        new_list = []
        for i in range(len_table):
            new_list.append(list_table[i][0])
        max_mode = max(new_list) # use the max value here
    return max_mode


def base(x):
    """
    Dummy function that does nothing

    :param x:
    :return:
    """
    return x

base.__name__ = 'base'