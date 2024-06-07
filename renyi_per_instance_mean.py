#This script computes the per-step per-instance Renyi DP guarantees for the mean update rule
import pandas as pd
import numpy as np
import scipy


res_file = "res/res.csv" # make sure to modify this to the correct sensitivity result file
feature = 'distance (mean)'
dataset = "MNIST"
model = "lenet"
cn = 1
eps = 10

df = pd.read_csv(res_file)
df = df[(df['model'] == model) & (df['eps'] == eps)]
assert len(df['sigma'].unique()) == 1
assert len(df['batch_size'].unique()) == 1
assert len(df['alpha'].unique()) == 1
sigma = df['sigma'].unique()[0]
bs = df['batch_size'].unique()[0]
alpha = df['alpha'].unique()[0]
if dataset == "MNIST":
    p = bs / 60000
elif dataset == "CIFAR10":
    p = bs / 50000
else:
    raise NotImplementedError

if df[feature].dtype != 'float64':
    df[feature] = df[feature].astype('float64')
df[feature] = df[feature] / (- 2 * (sigma) ** 2)
df = df.groupby(["point", 'batch'], as_index=False)[feature].apply(
    lambda grp: scipy.special.logsumexp(grp) - np.log(grp.count()))
df[feature] = df[feature] / (alpha - 1)
df = df.groupby(["point"], as_index=False)[feature].apply(
    lambda grp: np.average(grp))
df = df.rename(columns={feature: "Privacy cost"})
print(df)
