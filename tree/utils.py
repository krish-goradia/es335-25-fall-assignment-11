"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import math
from scipy.special import xlogy
import numpy as np



def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    new_encoded_df = pd.DataFrame(index = X.index)

    for col in X.columns:
        if(X[col].dtype.name == "object" or X[col].dtype.name == "category"):
            unique_val = X[col].dropna().unique()
            for val in unique_val:
                new_encoded_df[f"{col}_{val}"] = (X[col]==val).astype(int)
        else:
            new_encoded_df[col]=X[col]
            
    return new_encoded_df

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    y = y.dropna()
    if y.dtype.kind == 'f':
        return True

    
    if y.dtype.kind in ('i', 'u'):  
        unique_ratio = y.nunique() / len(y)
        return unique_ratio > 0.1

    
    return False
    


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    values, counts = np.unique(Y, return_counts=True)
    probabilities = counts / len(Y)
    return -np.sum(xlogy(probabilities,probabilities) / np.log(2))
    
   


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    values, counts = np.unique(Y, return_counts=True)
    probabilities = counts / len(Y)
    return 1 - np.sum(probabilities ** 2)



def information_gain(Y: pd.Series, attr: pd.Series, criterion: str, threshold=None) -> float:
    """
    Calculate information gain for a given attribute.
    If threshold is provided, splits attribute into <= threshold and > threshold (for numeric).
    If threshold is None, assumes categorical attribute.

    Returns: information gain value (float)
    """
    # Original impurity
    if criterion == "entropy":
        original_impurity = entropy(Y)
    elif criterion == "gini":
        original_impurity = gini_index(Y)
    elif criterion == "mse":
        original_impurity = mse(Y)
    else:
        raise ValueError("Invalid criterion.")

    weighted_impurity = 0

    if threshold is not None:  # Numeric attribute
        left_Y = Y[attr <= threshold]
        right_Y = Y[attr > threshold]

        if len(left_Y) == 0 or len(right_Y) == 0:
            return 0  # No valid split

        left_weight = len(left_Y) / len(Y)
        right_weight = len(right_Y) / len(Y)

        if criterion == "entropy":
            left_impurity = entropy(left_Y)
            right_impurity = entropy(right_Y)
        elif criterion == "gini":
            left_impurity = gini_index(left_Y)
            right_impurity = gini_index(right_Y)
        else:
            left_impurity = mse(left_Y)
            right_impurity = mse(right_Y)

        weighted_impurity = (left_weight * left_impurity) + (right_weight * right_impurity)

    else:  # Categorical attribute
        for value in attr.unique():
            subset_Y = Y[attr == value]
            weight = len(subset_Y) / len(Y)

            if criterion == "entropy":
                subset_impurity = entropy(subset_Y)
            elif criterion == "gini":
                subset_impurity = gini_index(subset_Y)
            else:
                subset_impurity = mse(subset_Y)

            weighted_impurity += weight * subset_impurity

    return original_impurity - weighted_impurity

    


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon

    """
    best_feature = None
    best_threshold = None
    best_gain = -float("inf")

    for feature in features:
        col = X[feature]

        if np.issubdtype(col.dtype, np.number):  # Continuous feature
            sorted_idx = col.argsort()
            sorted_values = col.iloc[sorted_idx]
            sorted_labels = y.iloc[sorted_idx]

            # Generate candidate thresholds where class changes
            candidates = []
            for i in range(1, len(sorted_values)):
                if sorted_labels.iloc[i] != sorted_labels.iloc[i-1]:
                    split = (sorted_values.iloc[i] + sorted_values.iloc[i-1]) / 2
                    candidates.append(split)

            # Check information gain for each threshold
            for threshold in candidates:
                gain = information_gain(y, col, criterion, threshold=threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        else:  # Categorical feature
            gain = information_gain(y, col, criterion)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = None

    return best_feature, best_threshold, best_gain


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    col = X[attribute]

    if np.issubdtype(col.dtype, np.number):  # Continuous attribute
        # value is a threshold
        left_mask = col <= value
        right_mask = col > value
    else:  # Categorical attribute
        # value is a category
        left_mask = col == value
        right_mask = col != value

    left_X = X[left_mask]
    left_y = y[left_mask]
    right_X = X[right_mask]
    right_y = y[right_mask]

    return left_X, left_y, right_X, right_y
    
