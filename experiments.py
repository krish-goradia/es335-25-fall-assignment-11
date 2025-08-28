import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 10  # Number of times to run each experiment to calculate the average values

def create_fake_data(N, P):
    X = pd.DataFrame({i: pd.Series(np.random.randint(2, size=N), dtype="category") for i in range(P)})
    y = pd.Series(np.random.randint(2, size=N), dtype="category")
    return X, y

def measure_time(N,P):
    X, y = create_fake_data(N,P)
    
    fit_time = []
    predict_time = []

    for _ in range(num_average_time):
        tree = DecisionTree("information_gain")

        start = time.time()
        tree.fit_disc_disc(X,y)
        mid = time.time()
        tree.predict(X)
        end = time.time()

        fit_time.append(mid-start)
        predict_time.append(end-mid)
        print(tree)
    
    return np.mean(fit_time), np.mean(predict_time)

def plot_results(x_values, fit_times, predict_times, x_label, title, P=None, N=None):
    # Compute theoretical complexities
    if P is not None:  # Varying N (P fixed)
        scale_train = fit_times[0] / (x_values[0] * np.log2(x_values[0]) * P)
        scale_predict = predict_times[0] / (x_values[0] * np.log2(x_values[0]))
        theoretical_train = [scale_train * (n * np.log2(n) * P) for n in x_values]
        theoretical_predict = [scale_predict * (n * np.log2(n)) for n in x_values]
    else:  # Varying P (N fixed)
        scale_train = fit_times[0] / (N * np.log2(N) * x_values[0])
        theoretical_train = [scale_train * (N * np.log2(N) * p) for p in x_values]
        theoretical_predict = [predict_times[0]] * len(x_values)  # Prediction ~ flat

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, fit_times, marker='o', label="Training Time (Experimental)")
    plt.plot(x_values, predict_times, marker='s', label="Prediction Time (Experimental)")
    plt.plot(x_values, theoretical_train, '--', label="O(N * P * log N) (Training)")
    plt.plot(x_values, theoretical_predict, '--', label="O(N log N) (Prediction)")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid()
    plt.show()

N_values = [10*i for i in range(2,11)]
P_fixed = 10
fit_times_N, predict_times_N = [], []

for N in N_values:
    fit_t, predict_t = measure_time(N, P_fixed)
    fit_times_N.append(fit_t)
    predict_times_N.append(predict_t)

plot_results(N_values, fit_times_N, predict_times_N, "Number of Samples (N)",
             f"Information Gain: Time vs N (P={P_fixed})", P=P_fixed)

# Experiment 2: Vary P (features), keep N fixed
P_values = [5*i for i in range(2,11)]
N_fixed = 1000
fit_times_P, predict_times_P = [], []

for P in P_values:
    fit_t, predict_t = measure_time(N_fixed, P)
    fit_times_P.append(fit_t)
    predict_times_P.append(predict_t)

plot_results(P_values, fit_times_P, predict_times_P, "Number of Features (P)",
             f"Information Gain: Time vs P (N={N_fixed})", N=N_fixed)