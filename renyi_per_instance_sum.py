#This script computes the per-step per-instance Renyi DP guarantees for the sum update rule
import pandas as pd
import numpy as np
import scipy
import math


def binom(n, k):
    return math.factorial(n) / math.factorial(k) / math.factorial(n - k)


def renyi_baseline(alpha, sigma, sample_rate, cn):
    res = []
    for k in range(alpha + 1):
        coeff = np.log(binom(alpha, k) * math.pow(1 - sample_rate, alpha - k) * math.pow(sample_rate, k))
        expect = math.pow(cn, 2) * k * (k - 1) / (2 * math.pow(sigma, 2))
        res.append(coeff + expect)
    return scipy.special.logsumexp(res) / (alpha - 1)


res_file = "res/res.csv" # make sure to modify this to the correct sensitivity result file
feature = 'distance (sum)'
dataset = "MNIST"
model = "lenet"
cn = 1
eps = 10
alpha = 8

df = pd.read_csv(res_file)
df = df[(df['model'] == model) & (df['eps'] == eps)]
assert len(df['sigma'].unique()) == 1
assert len(df['batch_size'].unique()) == 1
sigma = df['sigma'].unique()[0]
bs = df['batch_size'].unique()[0]
if dataset == "MNIST":
    p = bs / 60000
elif dataset == "CIFAR10":
    p = bs / 50000
else:
    raise NotImplementedError

df = df.groupby(["point"], as_index=False)[feature].mean()
#NOTE: multipling 'grad norm' by batch size as we actually saved the grad norm divided by the batch size
df[feature] = df[feature].apply(lambda x: renyi_baseline(alpha, sigma, p, x * bs))
df = df.rename(columns={feature: "Privacy cost"})
print(df)
