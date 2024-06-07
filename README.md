# Gradients Look Alike: Sensitivity is Often Overestimated in DP-SGD

This repository is an implementation of the paper [Gradients Look Alike: Sensitivity is Often Overestimated in DP-SGD](https://arxiv.org/abs/2307.00310), accepted to the 33rd USENIX Security
Symposium. In this paper we provided analysis for the per-instance guarantees of DP-SGD, giving a tool to explain the failure of privacy attacks on many datapoints (on common datasets) and also show the possibility of efficient unlearning for many points, amongst other implications. This was done by introducing new per-instance per-step analyses, and a new per-instance composition theorem.

We test our code on two datasets: MNIST, and CIFAR-10. 

### Dependency
Our code is implemented and tested on PyTorch. See `requirements.txt` for details.

### Compute Sensitivity
One must compute the sensitivity of certain data points before computing their per-instance guarantees of DP-SGD.

To compute the sensitivity of certain data points at different stages of training a model using DPSGD:
```
python compute_sensitivity.py --exp [eps_delta or renyi] --points [indices of data points in the dataset] --stage [training stage (0 to 1 where 0 means no training has been done and 1 means sensitivity at the last checkpoint)] --reduction [update rule, mean or sum] --res-name [sensitivity will be saved in res/[res-name].csv]
```
`exp` denotes whether to compute the epsilon-delta guarantee or Renyi guarantee. There are a few other arguments that you could find at the beginning of the script. 

Note that if `exp` is set to `renyi` while `reduction` is `sum`, we will use the eps_delta sensitivity for computing Renyi sensitivity (see Corollary 3.1/Theorem 3.2).


### Compute Privacy Cost
After sensitivity is computed, to compute per-instance guarantees of DP-SGD, one may choose one of the following.
Note we do not need to specify the point indices here as the scrips will compute the guarantees for all points whose
sensitivity has been computed.

For epsilon-delta guarantee:
```
python eps_delta.py
```

For Renyi guarantee (mean update rule)
```
python renyi_per_instance_mean.py
```

For Renyi guarantee (sum update rule)
```
python renyi_per_instance_sum.py
```

For (Renyi) composition guarantee (only applied to the sum update rule)
```
python renyi_per_instance_sum_compo.py
```
Note for the composition, one should compute the sensitivity for the points at multiple training stages (i.e., run 
`compute_sensitivity.py` with different `stage` and save the sensitivity in the same result file).

Please make sure to load the correct sensitivity result file when computing the guarantees.

### Questions or suggestions
If you have any questions or suggestions, feel free to raise an issue or reach out to us via email.
