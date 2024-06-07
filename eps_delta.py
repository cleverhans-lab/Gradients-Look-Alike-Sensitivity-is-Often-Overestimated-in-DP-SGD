import pandas as pd
import numpy as np
import scipy


def sensitivity_to_eps(log_sensitivity, p, p_x_star, delta_prime):
    t1 = (1 / p) * (1 / (1 - 1 / p)) * log_sensitivity
    t2 = 1 / (1 - 1 / p) * np.log(p_x_star)
    t3 = 1 / (1 - p) * np.log(delta_prime)
    res = np.log(np.exp(t1 + t2 + t3) + 1 - p_x_star)
    return res


res_file = "res/res.csv" # make sure to modify this to the correct sensitivity result file
feature = 'distance (sum)' # or distance (mean)
cn = 1
eps = 10
delta_prime_prime = 1e-5
p = 1e4

df = pd.read_csv(res_file)
sigma = df['sigma'].unique()[0]
p_x_star = df['p'].unique()[0]
p_curr = p
delta = (delta_prime_prime / 2) / p_x_star
c = np.sqrt(2 * np.log(1.25 / delta)) / sigma
df[feature] = df[feature] * p_curr * c
delta_prime = delta_prime_prime - delta * p_x_star
df = df.groupby(["point"], as_index=False)[feature].apply(lambda grp: scipy.special.logsumexp(grp) - np.log(grp.count()))
df[feature] = df[feature].apply(lambda log_sensitivity: sensitivity_to_eps(log_sensitivity, p_curr, p_x_star, delta_prime))
df = df.rename(columns={feature: "Privacy cost"})
print(df)
