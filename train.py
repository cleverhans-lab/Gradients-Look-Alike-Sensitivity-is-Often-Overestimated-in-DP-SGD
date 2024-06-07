import numpy as np
import os
import torch
import copy
import shutil
import inspect
import warnings
import torchvision
from opacus import PrivacyEngine, GradSampleModule
from opacus.accountants.utils import get_noise_multiplier
import utils
import model


torch.backends.cudnn.benchmark = True


class train_fn():
    def __init__(self, lr=0.01, batch_size=128, dataset='SVHN', architecture="resnet20", exp_id=None,
                 model_dir=None, save_freq=None, dec_lr=None, trainset=None, save_name=None, num_class=10,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), seed=0, optimizer="sgd",
                 gamma=0.1, overwrite=0, epochs=10, dp=0, sigma=None, cn=1, delta=1e-5, eps=1, norm_type='gn',
                 sample_data=1, poisson=False, remove_points=None, reduction="sum"):
        inputs = inspect.signature(train_fn).parameters
        for item in inputs:
            setattr(self, item, eval(item))

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.save_keyword = "model_step_"
        if save_name is None:
            save_name = f"ckpt_{self.dataset}_{architecture}_{int(eps) if eps % 1 == 0 else eps}_{exp_id}"

        try:
            architecture = eval(f"model.{architecture}")
        except:
            architecture = eval(f"torchvision.models.{architecture}")

        if save_freq is not None and save_freq > 0:
            self.save_dir = utils.get_save_dir(save_name)

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)
                print(f"mkdir {self.save_dir}")
            else:
                if len(os.listdir(self.save_dir)) > 0:
                    warnings.warn(f"Checkpointing directory is not empty {self.save_dir}")
                    if overwrite:
                        shutil.rmtree(self.save_dir)
                        os.mkdir(self.save_dir)
                        print(f"overwrite {self.save_dir}")
                        assert len(os.listdir(self.save_dir)) == 0
        else:
            self.save_dir = None

        if trainset is None:
            self.trainset = utils.load_dataset(self.dataset, True, download=True)
        else:
            self.trainset = trainset
        self.testset = utils.load_dataset(self.dataset, False, download=True)

        train_size = self.trainset.__len__()

        self.sequence = utils.create_sequences(batch_size, train_size, epochs, sample_data, poisson=poisson,
                                               remove_points=remove_points)

        if dataset == "MNIST":
            in_channel = 1
        else:
            in_channel = 3

        self.net = architecture(norm_type=norm_type, in_channels=in_channel)

        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,
                                                        shuffle=True, pin_memory=True)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size,
                                                      shuffle=False, pin_memory=True)

        num_batch = self.trainset.__len__() / self.batch_size

        self.net.to(self.device)
        self.optimizer, self.scheduler = utils.get_optimizer(dataset, self.net, lr, num_batch, dec_lr=dec_lr,
                                                             optimizer=optimizer, gamma=gamma)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        if dp:
            self.privacy_engine = PrivacyEngine()
            self.net, self.optimizer, _ = self.privacy_engine.make_private(
                module=self.net,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=get_noise_multiplier(
                    target_epsilon=self.eps,
                    target_delta=self.delta,
                    sample_rate=self.batch_size / train_size,
                    epochs=self.epochs,
                    accountant=self.privacy_engine.accountant.mechanism(),
                ),
                max_grad_norm=self.cn,
                loss_reduction=reduction,
            )
            self.sigma = self.optimizer.noise_multiplier
        else:
            self.privacy_engine = None

        if model_dir is not None:
            self.load(model_dir)

    def save(self, epoch=None, save_path=None):
        assert epoch is not None or save_path is not None
        if save_path is None:
            save_path = os.path.join(self.save_dir, f"model_step_{epoch + 1}")
        net_state_dict = self.net.state_dict()
        if not os.path.exists(save_path):
            state = {'net': net_state_dict,
                     'optimizer': self.optimizer.state_dict()}
            if self.scheduler is not None:
                state["scheduler"] = self.scheduler.state_dict()
            if self.privacy_engine is not None:
                state["privacy_engine_accountant"] = self.privacy_engine.accountant
            torch.save(state, save_path)

    def load(self, path):
        states = torch.load(path)
        self.net.load_state_dict(states['net'])
        self.optimizer.load_state_dict(states['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(states['scheduler'])
        if self.privacy_engine is not None:
            self.privacy_engine.accountant = states['privacy_engine_accountant']

    def predict(self, inputs):
        outputs = self.net(inputs)
        if isinstance(outputs, tuple) and len(outputs) == 1:
            outputs = outputs[0]
        elif len(outputs.shape) > 2:
            outputs = outputs.squeeze()
        elif not isinstance(outputs, torch.Tensor):
            outputs = outputs.logits
        return outputs

    def update(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step()

    def compute_loss(self, data):
        inputs, labels = data[0].to(self.device), data[1].to(self.device)
        outputs = self.predict(inputs.contiguous())
        loss = self.criterion(outputs, labels)
        return loss

    def train_step(self, data):
        loss = self.compute_loss(data)
        loss.backward()
        self.update()
        return loss.item()

    def train(self, step):
        self.net.train()
        if self.save_dir is not None:
            last_ckpt = utils.get_last_ckpt(self.save_dir, self.save_keyword)
            if last_ckpt > step + 1:
                return True
            elif last_ckpt == step + 1:
                print(f"loading checkpoints for {self.save_keyword}{last_ckpt} from {self.save_dir}")
                self.load(os.path.join(self.save_dir, f"{self.save_keyword}{last_ckpt}"))
                return True
        self.optimizer.zero_grad()
        indices = self.sequence[step]
        subset = torch.utils.data.Subset(self.trainset, indices)
        sub_trainloader = torch.utils.data.DataLoader(subset, batch_size=indices.shape[0])
        for batch_idx, data in enumerate(sub_trainloader, 0):
            self.train_step(data)
        assert batch_idx == 0

        if self.save_freq is not None and (step + 1) % self.save_freq == 0 and self.save_freq > 0:
            self.save(step)
        return False

    def validate(self):
        self.net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.predict(inputs.contiguous())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Test Accuracy: {100 * correct / total} %')

        return correct / total

    def compute_grad(self, data=None, indices=None, step=None, cn=-1):
        self.net.train()
        model_state = self.net.state_dict()
        if data is None:
            if indices is None:
                assert step is not None
                indices = self.sequence[step]
            subset = torch.utils.data.Subset(self.trainset, indices)
            sub_trainloader = torch.utils.data.DataLoader(subset, batch_size=indices.shape[0])
            for data in sub_trainloader:
                break
        batch_size = data[0].shape[0]
        inputs, labels = data[0].to(self.device), data[1].to(self.device)
        outputs = self.predict(inputs.contiguous())
        with torch.no_grad():
            correct = (torch.max(outputs.data, 1)[1] == labels).int().cpu().numpy()
        loss = self.criterion(outputs, labels)
        loss.backward()
        per_sample_grad = []
        for p in self.net.parameters():
            if hasattr(p, 'grad_sample'):
                per_sample_grad.append(p.grad_sample.detach().reshape([batch_size, -1]))
        per_sample_grad = torch.concat(per_sample_grad, 1)
        if cn >= 0:
            per_sample_norm = per_sample_grad.norm(2, dim=-1)
            per_sample_clip_factor = (cn / per_sample_norm).clamp(max=1.0).unsqueeze(-1)
            per_sample_grad = per_sample_grad * per_sample_clip_factor
        self.net.load_state_dict(model_state)
        self.optimizer.zero_grad()
        return per_sample_grad, correct

    def grad_to_sensitivity(self, per_sample_grad, batch_size, expected_batch_size):
        # compute the difference in gradient
        if self.reduction == 'mean':
            scale = (1 / batch_size - 1 / (batch_size - 1))
            sum_grad = torch.sum(per_sample_grad, 0, keepdim=True)
            res = torch.norm(scale * sum_grad + per_sample_grad / (batch_size - 1), p=2, dim=1)
            res = res.cpu().numpy()
        elif self.reduction == 'sum':
            res = torch.norm(per_sample_grad, p=2, dim=1) / expected_batch_size
            res = res.cpu().numpy()
        else:
            raise NotImplementedError(f"reduction strategy {self.reduction} is not recognized")
        return res

    def sensitivity(self, data=None, indices=None, step=None, cn=-1, expected_batch_size=0):
        # indices = [index of point interested, e.g., 0; random indices of a batch, e.g., 9, 4, 14, 90]
        # get the batch of data points
        if data is None:
            if indices is None:
                assert step is not None
                indices = self.sequence[step]
            batch_size = indices.shape[0]
        else:
            batch_size = data[0].shape[0]
        # compute per-sample gradient`
        per_sample_grad, correct = self.compute_grad(data, indices, step, cn)
        res = self.grad_to_sensitivity(per_sample_grad, batch_size, expected_batch_size)
        return res, correct

    def renyi_sen_eqn(self, g, gs, alpha):
        term1 = torch.sum(torch.pow(torch.norm(gs, p=2, dim=1), 2))
        term2 = (alpha - 1) * torch.pow(torch.norm(g, p=2), 2)
        term3 = torch.pow(torch.norm(torch.sum(gs, 0) - (alpha - 1) * g, p=2), 2)
        return term1 - term2 - term3

    def sensitivity_renyi(self, target_batch_index, alpha_batch_indices, alpha, cn=-1):
        # self.net = self.net.to_standard_module()
        # self.net = GradSampleModule(self.net)
        target_grad, _ = self.compute_grad(indices=target_batch_index, cn=cn)
        target_grad = target_grad.cpu()
        alpha_grads = []
        for batch_index in alpha_batch_indices:
            alpha_grads.append(torch.mean(self.compute_grad(indices=batch_index, cn=cn)[0], 0).cpu())
        res = []
        if self.reduction == 'mean':
            target_g = torch.mean(target_grad, 0)
            alpha_g = torch.stack(alpha_grads)
            res.append(self.renyi_sen_eqn(target_g, alpha_g, alpha).item())

        if self.reduction == 'sum':
            target_g = torch.sum(target_grad, 0)
            alpha_g = torch.stack([g * b.shape[0] for b, g in zip(alpha_batch_indices, alpha_grads)])
            res.append(self.renyi_sen_eqn(target_g, alpha_g, alpha).item())
        return res
