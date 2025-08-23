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
        self.threshold = None

    def fit_disc_disc(self, X: pd.DataFrame, y: pd.Series) -> None:

        feature = X.columns
        if len(set(y)) == 1 or self.depth == self.max_depth or X.shape[1] == 0:
            self.label = y.mode()[0]
            return
        
        bestFeature, _, bestGain = opt_split_attribute(X, y, self.criterion)

        if bestFeature is None:
            self.label = y.mode()[0]
            return
        
        self.feature = bestFeature
        
        col = X[bestFeature]

        self.branches = {}

        for val in col.unique():
            mask = col == val
            child = DecisionTree(self.criterion, self.max_depth, self.depth+1)
            self.branches[val] = child
            branch_X, branch_Y = X[mask], y[mask]
            child.fit_disc_disc(branch_X.drop(columns=[bestFeature]),branch_Y)


    def fit_disc_real(self, X: pd.DataFrame, y: pd.Series):
        feature = X.columns

        if len(set(y)) == 1 or self.depth == self.max_depth or X.shape[1] == 0:
            self.label = y.mean()  # regression leaf
            return

        best_feature, _, best_gain = opt_split_attribute(X, y, "mse")
        if best_feature is None:
            self.label = y.mean()
            return

        self.feature = best_feature
        col = X[best_feature]

        self.branches = {}

        for val in col.unique():
            mask = col == val
            child = DecisionTree(self.criterion, self.max_depth, self.depth+1)
            self.branches[val] = child
            branch_X, branch_Y = X[mask], y[mask]
            child.fit_disc_disc(branch_X.drop(columns=[best_feature]),branch_Y)
        
    def fit_real_disc(self, X: pd.DataFrame, y: pd.Series):
        feature = X.columns
        if len(set(y)) == 1 or self.depth == self.max_depth or X.columns == 0:
            self.label = y.mode()[0]
            return
        
        bestFeature, bestThreshold, bestGain = opt_split_attribute(X, y, self.criterion)

        if bestFeature is None:
            self.label = y.mode()[0]
            return
        
        self.feature = bestFeature
        col = X[bestFeature]

        self.left = DecisionTree(self.criterion,self.max_depth,self.depth+1)
        self.right = DecisionTree(self.criterion,self.max_depth,self.depth+1)



    def predict(self, X: pd.DataFrame) -> pd.Series:
        result = []
        for _,row in X.iterrows():
            result.append(self.predict_row(row))
        return pd.Series(result)

    def predict_row(self, row):
        if self.label is not None:
            return self.label
        
        if hasattr(self, 'branches') and self.branches is not None:
            value = row[self.feature]
            if value in self.branches:
                return  self.branches[value].predict_row(row)
            else:
                return self.label if self.label is not None else None
            
        if row[self.feature] <= self.threshold:
            return self.left.predict_row(row)
        else:
            return self.right.predict_row(row)

    def plot(self,indent = "") -> None:
        if self.label is not None:
            print(f"{self.label}")
            return

        if hasattr(self, 'branches') and self.branches:
            print(f"?({self.feature})")
            for val, subtree in self.branches.items():
                print(f"{indent}    Y({val}): ", end="")
                subtree.plot(indent + "    ")
        else:
            # Continuous split
            print(f"?({self.feature} > {self.threshold})")
            
            # Left branch → Yes
            print(f"{indent}    Y: ", end="")
            if self.left:
                if self.left.label is not None:
                    print(self.left.label)
                else:
                    self.left.plot(indent + "    ")
            else:
                print("None")
            
            # Right branch → No
            print(f"{indent}    N: ", end="")
            if self.right:
                if self.right.label is not None:
                    print(self.right.label)
                else:
                    self.right.plot(indent + "    ")
            else:
                print("None")
