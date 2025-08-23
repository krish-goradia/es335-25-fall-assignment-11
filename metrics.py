from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size, "y_hat and y must have equal length"
    if len(y) == 0:
        return 0.0
    
    correct_samples = (y_hat==y).sum()
    total_samples = len(y)

    return float(correct_samples/total_samples)


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size, "y_hat and y must have equal length"

    true_positive = ((y_hat==cls)& (y == cls)).sum()
    false_positive = ((y_hat==cls)& (y != cls)).sum()
    
    if true_positive + false_positive == 0:
        return 0.0
    return float(true_positive/(true_positive+false_positive))


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size, "y_hat and y must have equal length"

    true_positive = ((y_hat==cls)& (y == cls)).sum()
    false_negative = ((y_hat!=cls)&(y==cls)).sum()

    if true_positive + false_negative == 0:
        return 0.0
    return float(true_positive/(false_negative+true_positive))


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size, "y_hat and y must have equal length" 
    
    if len(y) == 0:
        return 0.0
    val_mse = ((y_hat-y)**2).mean()
    return float(np.sqrt(val_mse))


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size, "y_hat and y must have equal length" 
    
    if len(y) == 0:
        return 0.0
    val_mae = (np.abs((y_hat-y))).mean()
    return float(val_mae)
