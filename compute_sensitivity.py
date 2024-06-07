import torch
import argparse
import os
import pandas as pd
import numpy as np
import train
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--points', nargs="+", type=int, default=[0, 1], help='indices of data points to compute sensitivity')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--num-iters', type=int, default=20, help='only useful for renyi')
parser.add_argument('--alpha', type=int, default=8, help='only useful for renyi')
parser.add_argument('--num-batches', type=int, default=100, help='only useful for renyi')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--cn', type=float, default=1, help='clipping norm')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--dp', type=int, default=1)
parser.add_argument('--eps', type=float, default=10)
parser.add_argument('--optimizer', type=str, default="sgd")
parser.add_argument('--dataset', type=str, default="MNIST")
parser.add_argument('--model', type=str, default="lenet")
parser.add_argument('--norm-type', type=str, default="gn", help="Note that batch norm is not compatible with DPSGD")
parser.add_argument('--save-freq', type=int, default=100, help='frequence of saving checkpoints')
parser.add_argument('--save-name', type=str, default='ckpt', help='checkpoints will be saved under models/[save-name]')
parser.add_argument('--res-name', type=str, default='res', help='sensitivity will be saved in res/[res-name].csv')
parser.add_argument('--gamma', type=float, default=None, help='for learning rate schedule')
parser.add_argument('--dec-lr', nargs="+", type=int, default=None, help='for learning rate schedule')
parser.add_argument('--id', type=str, default='', help="experiment id")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--overwrite', type=int, default=0, help="whether overwrite existing result files")
parser.add_argument('--poisson-train', type=int, default=1, help="should always be 1 for correct DPSGD")
parser.add_argument('--stage', type=str, default='initial', help='initial, middle, final, or 0 to 1 where 0 means not'
                                                                 'training has beend done and 1 means training finishes')
parser.add_argument('--reduction', type=str, default='sum', help="update rule, mean or sum")
parser.add_argument('--exp', type=str, default='renyi', help='experiment type: eps_delta, or renyi')
parser.add_argument('--less-point', type=int, default=0, help="if set to 1, we consider the dataset with 1 less point."
                                                              "Note the missing point will impact how training, so"
                                                              "in this case arg.points can only contain 1 point.")
arg = parser.parse_args()
np.random.seed(arg.seed)

if arg.less_point:
    assert isinstance(arg.points, int)
    remove_point = arg.points
else:
    remove_point = None
if isinstance(arg.points, int):
    arg.points = [arg.points]
point_to_do = np.array(arg.points)

train_fn = train.train_fn(arg.lr, arg.batch_size, arg.dataset, arg.model,
                          exp_id=arg.id, save_freq=arg.save_freq, optimizer=arg.optimizer, epochs=arg.epochs,
                          dp=arg.dp, cn=arg.cn, eps=arg.eps, dec_lr=arg.dec_lr, gamma=arg.gamma, seed=arg.seed,
                          norm_type=arg.norm_type, poisson=arg.poisson_train, save_name=arg.save_name,
                          remove_points=remove_point, reduction=arg.reduction)
trainset_size = train_fn.trainset.__len__()
p = arg.batch_size / trainset_size
all_indices = np.arange(trainset_size)

if not os.path.exists("res"):
    os.mkdir("res")
res_dir = f"res/{arg.res_name}.csv"
temp_res_dir = res_dir.replace("res/", "res/temp_")
if os.path.exists(temp_res_dir):
    os.remove(temp_res_dir)
print(f"path to result file: {res_dir}")

step = utils.find_ckpt(arg.stage, trainset_size, arg.batch_size, arg.save_freq, arg.epochs)
cur_path = f"{train_fn.save_dir}/model_step_{step}"

###########
# the code block below checks if training is needed by looking for the checkpoints
if not os.path.exists(cur_path):
    print("checkpoints not found, starting training")
    train_fn.save(-1)
    for step in range(train_fn.sequence.shape[0]):
        train_fn.train(step)
    train_fn.validate()
    step = utils.find_ckpt(arg.stage, trainset_size, arg.batch_size, arg.save_freq, arg.epochs)
    cur_path = f"{train_fn.save_dir}/model_step_{step}"
    train_fn = train.train_fn(arg.lr, arg.batch_size, arg.dataset, arg.model,
                              exp_id=arg.id, save_freq=arg.save_freq, optimizer=arg.optimizer, epochs=arg.epochs,
                              dp=arg.dp, cn=arg.cn, eps=arg.eps, dec_lr=arg.dec_lr, gamma=arg.gamma, seed=arg.seed,
                              norm_type=arg.norm_type, poisson=arg.poisson_train, reduction=arg.reduction)

train_fn.load(cur_path)
accuracy = train_fn.validate()
###########

if os.path.exists(res_dir) and not arg.overwrite:
    temp_df = pd.read_csv(res_dir)
    if "renyi" in arg.exp and arg.reduction == "mean":
        temp_df = temp_df[(temp_df['type'] == arg.stage) & (temp_df['alpha'] == arg.alpha)]
    else:
        temp_df = temp_df[temp_df['type'] == arg.stage]
    if temp_df.shape[0] != 0:
        tested_points = temp_df["point"].unique()
        points_list = [point for point in tested_points]
        start = len(points_list)
        point_to_do = list(set(point_to_do) - set(points_list))
        print(f"result file is not empty, found {start} points, {len(point_to_do)} points to analyze")
    else:
        print(f"{len(point_to_do)} points to analyze")
else:
    print(f"{len(point_to_do)} points to analyze")
    if arg.overwrite and os.path.exists(res_dir):
        os.remove(res_dir)

if len(point_to_do) == 0:
    print("skipped since results exist, set --overwrite 1 to re-run the experiment")
elif "renyi" in arg.exp and arg.reduction == "mean":
    # when using mean update rule and Renyi, we'll use Theorem 3.6
    for point_index in point_to_do:
        print(f"alpha = {arg.alpha}")
        # X, the dataset with 1 less point; all_indices is for X'
        remove1_indices = np.delete(all_indices, point_index)

        if "reverse" in arg.exp:
            size1, size2 = trainset_size, trainset_size - 1
            indices1, indices2 = all_indices, remove1_indices
        else:
            size1, size2 = trainset_size - 1, trainset_size
            indices1, indices2 = remove1_indices, all_indices

        for b in range(arg.num_iters):
            np.random.shuffle(indices1)
            np.random.shuffle(indices2)

            # target_batch is x_B from X
            sampling = np.random.binomial(1, p, size1)
            target_batch = indices1[sampling.astype(np.bool8)]

            # these are the num_batches of alpha batches from X', size [num_batches, arg.alpha * batch_size]
            alpha_batches = []
            for i in range(arg.num_batches):
                alpha_batches.append([])
                for j in range(arg.alpha):
                    sampling = np.random.binomial(1, p, size2)
                    alpha_batches[-1].append(indices2[sampling.astype(np.bool8)])

            for batch in alpha_batches:
                res = train_fn.sensitivity_renyi(target_batch, batch, arg.alpha, cn=arg.cn)

                if os.path.exists(temp_res_dir):
                    df = pd.read_csv(temp_res_dir)
                elif os.path.exists(res_dir):
                    df = pd.read_csv(res_dir)
                else:
                    df = pd.DataFrame()
                df = pd.concat([df, pd.DataFrame({f"distance ({arg.reduction})": res[0], "step": step, "p": p, "batch": b,
                                "point": point_index, "sigma": train_fn.sigma, "accuracy": accuracy, **vars(arg)})])
                df.to_csv(temp_res_dir, index=False)
                torch.cuda.empty_cache()

        os.rename(temp_res_dir, res_dir)
        torch.cuda.empty_cache()
else:
    # 1. For epsilon-delta
    # 2. OR use eps_delta for renyi, only applicable to the sum update rule (see Corollary 3.1/Theorem 3.2)
    indices = point_to_do
    real_bs = len(point_to_do)
    res, correct = train_fn.sensitivity(indices=indices, cn=arg.cn, expected_batch_size=arg.batch_size)

    if os.path.exists(temp_res_dir):
        df = pd.read_csv(temp_res_dir)
    elif os.path.exists(res_dir):
        df = pd.read_csv(res_dir)
    else:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame({f"distance ({arg.reduction})": res, "step": step,
                                      "real batch size": real_bs, "p": p, "point": indices, "sigma": train_fn.sigma,
                                      "correct": correct, "accuracy": accuracy, **vars(arg)})])
    df.to_csv(temp_res_dir, index=False)
    os.rename(temp_res_dir, res_dir)
