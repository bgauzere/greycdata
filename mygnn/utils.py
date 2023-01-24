import numpy as np


def nb_correct_classif(y_gt, y_pred):
    return int((y_gt == y_pred).sum())


def mse(y_gt, y_pred):
    """
    Compute the sum of squared errors between gt and pred
    """
    return np.mean((y_gt-y_pred)**2)


def rmse(y_gt, y_pred):
    return np.sqrt(mse(y_gt, y_pred))
