"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5, depth=0):
        self.criterion = criterion
        self.max_depth = max_depth
        self.depth = depth
        self.left = None
        self.right = None
        self.label = None
        self.feature = None
        self.value = None

    def fit_disc_disc(self, X: pd.DataFrame, y: pd.Series) -> None:
        if(len(set(y)) == 1 or self.depth == self.max_depth):
            self.label = y.mode()[0]
            return
        
        bestFeature, bestScore = None, float('-inf')
        bestSplits = None

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        pass

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        pass
