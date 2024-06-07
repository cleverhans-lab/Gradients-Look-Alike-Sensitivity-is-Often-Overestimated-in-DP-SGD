import torchvision
import os
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms


def create_sequences(batch_size, dataset_size, epochs, sample_data, poisson=False, remove_points=None):
    # create a sequence of data indices used for training
    num_batch = (epochs * dataset_size) // batch_size
    dataset = np.arange(dataset_size)
    if remove_points is not None:
        if not isinstance(remove_points, list):
            remove_points = [remove_points]
        for remove_point in remove_points:
            dataset = dataset[dataset != remove_point]
        dataset_size = dataset.shape[0]
    if sample_data < 1:
        sample_vector = np.random.default_rng().choice([False, True], size=dataset_size, replace=True,
                                                       p=[1 - sample_data, sample_data])
        dataset = dataset[sample_vector]
        dataset_size = dataset.shape[0]
    if poisson:
        p = batch_size / dataset_size
        sequence = []
        for _ in range(num_batch):
            sampling = np.random.binomial(1, p, dataset_size)
            indices = dataset[sampling.astype(np.bool8)]
            sequence.append(indices)
        sequence = np.array(sequence, dtype=object)
    else:
        sequence = np.concatenate([np.random.default_rng().choice(dataset, size=dataset_size, replace=False)
                                   for i in range(epochs)])
        sequence = np.reshape(sequence[:num_batch * batch_size], [num_batch, batch_size])
    return sequence


def load_dataset(dataset, train, download=False, apply_transform=False):
    try:
        dataset_class = eval(f"torchvision.datasets.{dataset}")
    except:
        raise NotImplementedError(f"Dataset {dataset} is not implemented by pytorch.")

    if dataset == "MNIST":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset == "FashionMNIST":
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])
    elif dataset == "CIFAR100":
        if train and apply_transform:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
                transforms.Resize([224, 224])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
                transforms.Resize([224, 224])
            ])
    else: #CIFAR10
        if train and apply_transform:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    try:
        data = dataset_class(root='./data', train=train, download=download, transform=transform)
    except:
        if train:
            data = dataset_class(root='./data', split="train", download=download, transform=transform)
        else:
            data = dataset_class(root='./data', split="test", download=download, transform=transform)
    return data


def get_optimizer(dataset, net, lr, num_batch, dec_lr=None, privacy_engine=None, gamma=0.1, optimizer="sgd"):
    if dataset == 'MNIST' and optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=lr)
        scheduler = None
    elif dataset == 'CIFAR10' and optimizer == "sgd":
        if dec_lr is None:
            dec_lr = [100, 150]
        if gamma is None:
            gamma = 0.1
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[round(i * num_batch) for i in dec_lr],
                                                   gamma=gamma)
    elif dataset == 'CIFAR100' and optimizer == "sgd":
        if dec_lr is None:
            dec_lr = [60, 120, 160]
        if gamma is None:
            gamma = 0.2
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[round(i * num_batch) for i in dec_lr],
                                                   gamma=gamma)
    elif optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=lr)
        scheduler = None
    else:
        print("using adam")
        optimizer = optim.Adam(net.parameters(), lr=lr)
        scheduler = None
    if privacy_engine is not None:
        privacy_engine.attach(optimizer)
    return optimizer, scheduler


def get_save_dir(save_name):
    if not os.path.exists("models"):
        os.mkdir("models")
    return os.path.join("models", save_name)


def find_ckpt(ckpt_step, trainset_size, batch_size, save_freq, epochs):
    if isinstance(ckpt_step, str):
        if ckpt_step == "final":
            ckpt_step = 1
        elif ckpt_step == "middle":
            ckpt_step = 0.5
        elif ckpt_step == "initial":
            ckpt_step = 0
        else:
            ckpt_step = float(ckpt_step)

    total_ckpts = trainset_size * epochs // batch_size // save_freq
    return round(total_ckpts * ckpt_step) * save_freq


def get_last_ckpt(save_dir, keyword):
    saved_points = [int(model_path[len(keyword):]) for model_path in os.listdir(save_dir)
                    if keyword in model_path]
    return max(saved_points)
