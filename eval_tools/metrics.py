import numpy as np

import statistics
from typing import Tuple

def get_overlap_and_exclusive(target: list, proposal: list) -> Tuple[list, list, list]:
    """
    Get the intersect of 2 sets as well as tokens
    missed by proposal and extra tokens in proposal
    """
    matched = np.intersect1d(target, proposal)
    extra = np.setxor1d(matched, proposal)
    missed = np.setxor1d(matched, target)
    return matched, extra, missed

def get_jaccard_idx(x: set, y: set) -> float:
    """
    Returns the Jaccard index of 2 integer sets
    """
    inter = x.intersection(y)
    union = x.union(y)

    return len(inter)/len(union)

def get_prec_rec_f1(target: list, proposal: list) -> Tuple[float, float, float]:
    """
    Function to return precision, recall and 
    f1 score for a given target and proposal
    """
    matched, extra, missed = get_overlap_and_exclusive(target, proposal)
    # Precision (punished for extra tokens predicted)
    precision = len(matched) / (len(matched) + len(extra))
    # Recall (Punished for tokens missed in prediction)
    recall = len(matched) / (len(matched) + len(missed))
    # F1 score (harmonic mean of precision/recall)
    f1_score = statistics.harmonic_mean([precision, recall])

    return precision, recall, f1_score