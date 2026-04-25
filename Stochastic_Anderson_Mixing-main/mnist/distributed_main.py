from __future__ import print_function

import argparse
import math
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import pinverse as pinv
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def take_subset(loader):
    subset = None
    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx > 0:
            return subset
        subset = [(d, t) for d, t in zip(data, target)]
    return subset


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def split_dataset(part_dataset, num_workers, partition, seed):
    if partition == "round_robin":
        shards = [[] for _ in range(num_workers)]
        for idx, item in enumerate(part_dataset):
            shards[idx % num_workers].append(item)
        return shards

    rng = random.Random(seed)
    items = list(part_dataset)
    rng.shuffle(items)
    shards = [[] for _ in range(num_workers)]
    shard_size = len(items) // num_workers
    extra = len(items) % num_workers
    offset = 0
    for worker_id in range(num_workers):
        take = shard_size + (1 if worker_id < extra else 0)
        shards[worker_id] = items[offset:offset + take]
        offset += take
    return shards


def flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()], dim=0)


def set_flat_params(model, flat_vector):
    offset = 0
    with torch.no_grad():
        for param in model.parameters():
            numel = param.numel()
            param.copy_(flat_vector[offset:offset + numel].view_as(param))
            offset += numel


def flat_grads(model):
    grads = []
    for param in model.parameters():
        if param.grad is None:
            grads.append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))
        else:
            grads.append(param.grad.view(-1))
    return torch.cat(grads, dim=0)


def evaluate(model, device, test_loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_num = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            total_num += len(data)
    return total_loss / total_num, total_correct / total_num


class DistributedSAMServer:
    """Center-side SAM specialized to distributed gradient aggregation."""

    def __init__(self, dim, device, dtype, args):
        self.alg = args.alg
        self.lr = args.lr
        self.beta = args.sam_beta
        self.alpha = args.sam_alpha
        self.damp = args.sam_damp
        self.hist_length = args.sam_hist_length
        self.period = args.sam_period
        self.gamma = args.sam_gamma
        self.eps = 1e-8
        self.counts = {
            "accepted": 0,
            "fallback": 0,
            "period_skip": 0,
            "insufficient_history": 0,
        }

        self.step = 0
        self.Xk = torch.zeros((dim, self.hist_length), device=device, dtype=dtype)
        self.Rk = torch.zeros((dim, self.hist_length), device=device, dtype=dtype)
        self.x_prev = torch.zeros(dim, device=device, dtype=dtype)
        self.res_prev = torch.zeros(dim, device=device, dtype=dtype)
        self.d_x_avg = torch.zeros(dim, device=device, dtype=dtype)
        self.d_res_avg = torch.zeros(dim, device=device, dtype=dtype)

    def update(self, xk, avg_grad):
        # res_k is the center descent step induced by the averaged worker gradient.
        res = (-self.lr * avg_grad).to(xk.dtype)

        if self.alg == "dsgd":
            self.x_prev.copy_(xk)
            self.res_prev.copy_(res)
            self.step += 1
            return xk + res

        cnt = self.step
        has_history = cnt > 0
        if has_history:
            self.d_x_avg.mul_(self.gamma).add_(xk - self.x_prev, alpha=1 - self.gamma)
            self.d_res_avg.mul_(self.gamma).add_(res - self.res_prev, alpha=1 - self.gamma)
            k = (cnt - 1) % self.hist_length
            self.Xk[:, k] = self.d_x_avg
            self.Rk[:, k] = self.d_res_avg
            delta_x = self.Xk[:, k]

        self.x_prev.copy_(xk)
        self.res_prev.copy_(res)

        x_next = xk + res
        if not has_history:
            self.counts["insufficient_history"] += 1
            self.step += 1
            return x_next

        if cnt % self.period != 0:
            self.counts["period_skip"] += 1
            self.step += 1
            return x_next

        delta = self.damp * res.dot(res) / (delta_x.dot(delta_x) + self.eps)
        gram = (self.Rk.t() @ self.Rk) + delta * (self.Xk.t() @ self.Xk)
        gamma_k = pinv(gram.double()).to(xk.dtype) @ (self.Rk.t() @ res)
        x_delta = self.beta * res - (self.alpha * self.Xk + self.alpha * self.beta * self.Rk) @ gamma_k

        # Keep the same simple acceptance spirit as the original SAM code.
        if torch.isfinite(x_delta).all() and x_delta.dot(res) > 0:
            x_next = xk + x_delta
            self.counts["accepted"] += 1
        else:
            self.counts["fallback"] += 1

        self.step += 1
        return x_next


def main():
    parser = argparse.ArgumentParser(description="Distributed MNIST experiment for SAM")
    parser.add_argument("--train-part-size", type=int, default=12000)
    parser.add_argument("--test-part-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=600)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--partition", type=str, default="iid", choices=["iid", "round_robin"])
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--dump-data", default="distributed_output.pkl", type=str)
    parser.add_argument("--alg", type=str, default="dsam", choices=["dsgd", "dsam"])
    parser.add_argument("--sam-period", type=int, default=1)
    parser.add_argument("--sam-hist-length", type=int, default=10)
    parser.add_argument("--sam-alpha", type=float, default=1.0)
    parser.add_argument("--sam-beta", type=float, default=1.0)
    parser.add_argument("--sam-damp", type=float, default=1e-2)
    parser.add_argument("--sam-gamma", type=float, default=0.9)
    parser.add_argument("--precision", type=int, default=1, choices=[0, 1])
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    set_seed(args.seed)

    part_kwargs = {"batch_size": args.train_part_size, "shuffle": True}
    test_part_kwargs = {"batch_size": args.test_part_size, "shuffle": True}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        part_kwargs.update(cuda_kwargs)
        test_part_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset_train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    dataset_test = datasets.MNIST("./data", train=False, transform=transform)
    part_train_loader = torch.utils.data.DataLoader(dataset_train, **part_kwargs)
    part_test_loader = torch.utils.data.DataLoader(dataset_test, **test_part_kwargs)
    train_subset = take_subset(part_train_loader)
    test_subset = take_subset(part_test_loader)

    worker_shards = split_dataset(train_subset, args.num_workers, args.partition, args.seed)
    worker_loaders = []
    for shard in worker_shards:
        worker_loaders.append(
            torch.utils.data.DataLoader(
                shard,
                batch_size=args.batch_size,
                shuffle=True,
            )
        )
    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=args.test_batch_size,
        shuffle=False,
    )

    model = Net().to(device)
    dtype = torch.float64 if args.precision == 1 else next(model.parameters()).dtype
    dim = flat_params(model).numel()
    server = DistributedSAMServer(dim=dim, device=device, dtype=dtype, args=args)
    results = {"train_loss": [], "train_prec": [], "test_loss": [], "test_prec": [], "counts": []}

    steps_per_epoch = max(1, math.ceil(len(train_subset) / max(args.num_workers * args.batch_size, 1)))

    for epoch in range(1, args.epochs + 1):
        model.train()
        shard_iters = [iter(loader) for loader in worker_loaders]
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for step_idx in range(steps_per_epoch):
            xk = flat_params(model).to(dtype)
            grad_accum = torch.zeros(dim, device=device, dtype=dtype)

            for worker_idx, loader in enumerate(worker_loaders):
                try:
                    data, target = next(shard_iters[worker_idx])
                except StopIteration:
                    shard_iters[worker_idx] = iter(loader)
                    data, target = next(shard_iters[worker_idx])
                data, target = data.to(device), target.to(device)

                model.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                grad_accum.add_(flat_grads(model).to(dtype))

                epoch_loss += float(loss.item()) * len(data)
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_total += len(data)

            avg_grad = grad_accum / float(args.num_workers)
            x_next = server.update(xk, avg_grad)
            set_flat_params(model, x_next.to(next(model.parameters()).dtype))

            if step_idx % args.log_interval == 0:
                print(
                    "Epoch {} Step {}/{} | alg={} | accepted={} fallback={}".format(
                        epoch,
                        step_idx,
                        steps_per_epoch,
                        args.alg,
                        server.counts["accepted"],
                        server.counts["fallback"],
                    )
                )

        train_loss = epoch_loss / epoch_total
        train_prec = epoch_correct / epoch_total
        print(
            "Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                train_loss,
                epoch_correct,
                epoch_total,
                100.0 * train_prec,
            )
        )
        test_loss, test_prec = evaluate(model, device, test_loader)
        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                int(test_prec * len(test_subset)),
                len(test_subset),
                100.0 * test_prec,
            )
        )

        results["train_loss"].append(float(train_loss))
        results["train_prec"].append(float(train_prec))
        results["test_loss"].append(float(test_loss))
        results["test_prec"].append(float(test_prec))
        results["counts"].append(dict(server.counts))

    dump_path = Path(args.dump_data)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    with dump_path.open("wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
