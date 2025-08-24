import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from metrics import *
from tree.base import *
from tree.utils import mse

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn

data.replace("?",np.nan, inplace = True)
data.dropna(inplace = True)
y = data["mpg"]
X = data.drop(labels = ["mpg","car name"], axis = 1)

X = X.astype("float64")

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 42)

decision_tree= DecisionTree(criterion = "information_gain")

X_train = pd.DataFrame(X_train).reset_index(drop = True)
y_train = pd.Series(y_train).reset_index(drop = True)
y_test = pd.Series(y_test).reset_index(drop = True)
X_test = pd.DataFrame(X_test).reset_index(drop = True)

decision_tree.fit_real_real(X_train,y_train)
y_hat = decision_tree.predict(X_test)
y_hat = pd.Series(y_hat)


rmse_val = rmse(y_hat,y_test)
mae_val = mae(y_hat,y_test)
print( f"Mean Absolute Error :{mae_val:.2f}")
print( f"Root Mean Square Error :{rmse_val:.2f}")
x_axis_val = np.linspace(1,len(y_hat),len(y_hat))
plt.figure(figsize=(12, 8))
plt.plot(x_axis_val, y_test, color="blue", label="Testing Data")
plt.plot(x_axis_val, y_hat, color="red", label="Predicted Data")
plt.legend()
plt.show()






# Part 2 of Question 
data.replace("?",np.nan, inplace = True)
data.dropna(inplace = True)
y = data["mpg"]
X = data.drop(labels = ["mpg","car name"], axis = 1)

X = X.astype("float64")

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 42)
X_train = pd.DataFrame(X_train).reset_index(drop = True)
y_train = pd.Series(y_train).reset_index(drop = True)
y_test = pd.Series(y_test).reset_index(drop = True)
X_test = pd.DataFrame(X_test).reset_index(drop = True)

start_time = time.time()
decision_tree= DecisionTree(criterion = "information_gain")
decision_tree.fit_real_real(X_train,y_train)
y_hat_ours = decision_tree.predict(X_test)
end_time = time.time()
y_hat_ours = pd.Series(y_hat_ours)
print(f"Time taken to train and test our model with data: {(end_time-start_time):.2f}")


start_time = time.time()
reg = DecisionTreeRegressor()
reg.fit(X_train,y_train)
y_hat_inbuilt = reg.predict(X_test)
end_time = time.time()
y_hat_inbuilt = pd.Series(y_hat_inbuilt)
print(f"Time taken to train and test inbuilt model with data: {(end_time-start_time):.2f}")

print(f"Root Mean Squared Error by inbuilt model: {rmse(y_hat_inbuilt,y_test)}")
print(f"Root Mean Squared Error by our model: {rmse(y_hat_ours,y_test)}")





