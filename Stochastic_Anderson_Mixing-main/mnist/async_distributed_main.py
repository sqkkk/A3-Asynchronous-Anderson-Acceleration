from __future__ import print_function

import argparse
import datetime
import hashlib
import heapq
import json
import math
import os
import pickle
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import pinverse as pinv
from torch.utils.data import Subset
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resnet.resnet import BasicBlock as ProperCIFARBasicBlock
from resnet.resnet import ResNet as ProperCIFARResNet
from resnet.resnet import resnext29_4x24d as ProperCIFARResNeXt29_4x24d
from resnet.resnet import resnext29_8x16d as ProperCIFARResNeXt29_8x16d
from resnet.resnet import resnext29_16x8d as ProperCIFARResNeXt29_16x8d
from resnet.resnet import resnext29_8x64d as ProperCIFARResNeXt29_8x64d

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_reproducibility_metadata(args, device):
    source_path = Path(__file__).resolve()
    tracked_env = {
        key: os.environ.get(key)
        for key in (
            "ASCEND_RT_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "PYTHONHASHSEED",
            "CONDA_DEFAULT_ENV",
        )
    }
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_npu_version": getattr(torch_npu, "__version__", None) if torch_npu is not None else None,
        "device": str(device),
        "env": tracked_env,
        "source_file": str(source_path),
        "source_sha256": file_sha256(source_path),
        "dump_data": args.dump_data,
    }


def update_summary(results):
    if results["test_prec"]:
        results["summary"] = {
            "final_acc": float(results["test_prec"][-1]),
            "best_acc": float(max(results["test_prec"])),
            "final_loss": float(results["test_loss"][-1]),
            "best_loss": float(min(results["test_loss"])),
        }
    else:
        results["summary"] = {}


def write_result_artifacts(dump_path, results, partial=False):
    dump_path = Path(dump_path)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    if partial:
        pkl_path = dump_path.with_suffix(dump_path.suffix + ".partial.pkl")
        meta_path = dump_path.with_suffix(dump_path.suffix + ".partial.meta.json")
    else:
        pkl_path = dump_path
        meta_path = dump_path.with_suffix(dump_path.suffix + ".meta.json")

    with pkl_path.open("wb") as f:
        pickle.dump(results, f)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": results["args"],
                "reproducibility": results["reproducibility"],
                "summary": results.get("summary", {}),
                "status": results.get("status", {}),
            },
            f,
            indent=2,
            sort_keys=True,
        )


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
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


class CIFARNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFARNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(8, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(8, planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetCIFAR, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(8, 64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride_val in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride=stride_val))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class LogSoftmaxModel(nn.Module):
    def __init__(self, backbone):
        super(LogSoftmaxModel, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        return F.log_softmax(self.backbone(x), dim=1)


def replace_batchnorm_with_groupnorm(module, groups=8):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            gn_groups = min(groups, child.num_features)
            while child.num_features % gn_groups != 0 and gn_groups > 1:
                gn_groups -= 1
            replacement = nn.GroupNorm(gn_groups, child.num_features, affine=True)
            if child.affine:
                with torch.no_grad():
                    replacement.weight.copy_(child.weight)
                    replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            replace_batchnorm_with_groupnorm(child, groups=groups)
    return module


def build_model(dataset_name, model_name):
    if dataset_name in ("cifar10", "cifar100"):
        num_classes = 10 if dataset_name == "cifar10" else 100
        if model_name in ("auto", "resnet18"):
            return ResNetCIFAR(num_classes=num_classes)
        imported_cifar_factories = {
            "resnet20": lambda: ProperCIFARResNet(ProperCIFARBasicBlock, [3, 3, 3], num_classes=num_classes),
            "resnet32": lambda: ProperCIFARResNet(ProperCIFARBasicBlock, [5, 5, 5], num_classes=num_classes),
            "resnet44": lambda: ProperCIFARResNet(ProperCIFARBasicBlock, [7, 7, 7], num_classes=num_classes),
            "resnet56": lambda: ProperCIFARResNet(ProperCIFARBasicBlock, [9, 9, 9], num_classes=num_classes),
            "resnet110": lambda: ProperCIFARResNet(ProperCIFARBasicBlock, [18, 18, 18], num_classes=num_classes),
            "resnet1202": lambda: ProperCIFARResNet(ProperCIFARBasicBlock, [200, 200, 200], num_classes=num_classes),
            # CIFAR-style ResNeXt keeps the 32x32 stem instead of using the
            # heavier ImageNet ResNeXt-50 stem.
            "resnext29_4x24d": lambda: ProperCIFARResNeXt29_4x24d(num_classes=num_classes),
            "resnext29_8x16d": lambda: ProperCIFARResNeXt29_8x16d(num_classes=num_classes),
            "resnext29_16x8d": lambda: ProperCIFARResNeXt29_16x8d(num_classes=num_classes),
            "resnext29_8x64d": lambda: ProperCIFARResNeXt29_8x64d(num_classes=num_classes),
        }
        if model_name in imported_cifar_factories:
            backbone = imported_cifar_factories[model_name]()
            # The imported CIFAR ResNets use BatchNorm running buffers, but the
            # async scaffold only synchronizes tensor parameters between worker
            # snapshots and the server. Converting them to GroupNorm keeps the
            # CIFAR ResNet/ResNeXt topology while avoiding hidden state drift.
            backbone = replace_batchnorm_with_groupnorm(backbone, groups=8)
            # The imported CIFAR ResNet returns raw logits, while the async
            # training scaffold uses F.nll_loss everywhere. Wrap it so all
            # models expose log-probabilities consistently.
            return LogSoftmaxModel(backbone)
        return CIFARNet(num_classes=num_classes)
    return MNISTNet()


def build_dataset(dataset_name, cifar_augment="basic", random_erasing=0.0):
    if dataset_name in ("cifar10", "cifar100"):
        if dataset_name == "cifar10":
            dataset_cls = datasets.CIFAR10
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        else:
            dataset_cls = datasets.CIFAR100
            mean = (0.5071, 0.4867, 0.4408)
            std = (0.2675, 0.2565, 0.2761)

        train_steps = []
        if cifar_augment != "none":
            train_steps.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        if cifar_augment == "randaugment":
            train_steps.append(transforms.RandAugment(num_ops=2, magnitude=9))
        elif cifar_augment == "autoaugment":
            train_steps.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
        elif cifar_augment == "trivial":
            train_steps.append(transforms.TrivialAugmentWide())
        elif cifar_augment not in ("none", "basic"):
            raise ValueError(f"unknown cifar_augment: {cifar_augment}")
        train_steps.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        if random_erasing > 0:
            train_steps.append(transforms.RandomErasing(p=random_erasing, value="random"))
        train_transform = transforms.Compose(train_steps)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset_train = dataset_cls("./data", train=True, download=True, transform=train_transform)
        dataset_test = dataset_cls("./data", train=False, download=True, transform=test_transform)
        return dataset_train, dataset_test

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset_train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    dataset_test = datasets.MNIST("./data", train=False, transform=transform)
    return dataset_train, dataset_test


def make_random_subset(dataset, subset_size, seed):
    if subset_size >= len(dataset):
        return dataset
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return Subset(dataset, indices[:subset_size])


def dataset_label(dataset, idx):
    if isinstance(dataset, Subset):
        return dataset_label(dataset.dataset, dataset.indices[idx])
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return int(targets[idx])
    _, target = dataset[idx]
    return int(target)


def repair_minimum_shard_size(shards, min_size, seed):
    if min_size <= 0:
        return shards
    sizes = [len(shard) for shard in shards]
    if sum(sizes) < len(shards) * min_size:
        raise ValueError("dirichlet_min_size is too large for the requested split")
    rng = random.Random(seed)
    while min(sizes) < min_size:
        small = min(range(len(shards)), key=lambda idx: sizes[idx])
        donors = [idx for idx, size in enumerate(sizes) if size > min_size]
        if not donors:
            break
        large = max(donors, key=lambda idx: sizes[idx])
        move_pos = rng.randrange(len(shards[large]))
        moved = shards[large].pop(move_pos)
        shards[small].append(moved)
        sizes[large] -= 1
        sizes[small] += 1
    return shards


def dirichlet_split_dataset(part_dataset, num_workers, seed, alpha=0.05, min_size=1):
    if alpha <= 0:
        raise ValueError("dirichlet_alpha must be positive")

    label_buckets = {}
    for idx in range(len(part_dataset)):
        label = dataset_label(part_dataset, idx)
        label_buckets.setdefault(label, []).append(idx)

    rng = np.random.default_rng(seed)
    shards = [[] for _ in range(num_workers)]
    alpha_vec = np.full(num_workers, alpha, dtype=np.float64)

    for label in sorted(label_buckets):
        label_indices = np.asarray(label_buckets[label], dtype=np.int64)
        rng.shuffle(label_indices)
        proportions = rng.dirichlet(alpha_vec)
        split_points = (np.cumsum(proportions) * len(label_indices)).astype(int)[:-1]
        splits = np.split(label_indices, split_points)
        for worker_id, split in enumerate(splits):
            if len(split) > 0:
                shards[worker_id].extend(split.tolist())

    for shard in shards:
        rng.shuffle(shard)
    shards = repair_minimum_shard_size(shards, min_size=min_size, seed=seed + 97)
    for shard in shards:
        rng.shuffle(shard)
    return shards


def set_seed(seed, device_type="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    if device_type == "npu" and hasattr(torch, "npu"):
        torch.npu.manual_seed_all(seed)


def resolve_device(device_name):
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device_name == "npu":
        if torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("NPU was requested but torch_npu / torch.npu is not available.")
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch_npu is not None and hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    return torch.device("cpu")


def split_dataset(part_dataset, num_workers, partition, seed, dirichlet_alpha=0.05, dirichlet_min_size=1):
    all_indices = list(range(len(part_dataset)))
    if partition == "round_robin":
        shards = [[] for _ in range(num_workers)]
        for idx in all_indices:
            shards[idx % num_workers].append(idx)
        return shards
    if partition == "label_sorted":
        items = sorted(all_indices, key=lambda idx: dataset_label(part_dataset, idx))
        shards = [[] for _ in range(num_workers)]
        shard_size = len(items) // num_workers
        extra = len(items) % num_workers
        offset = 0
        for worker_id in range(num_workers):
            take = shard_size + (1 if worker_id < extra else 0)
            shards[worker_id] = items[offset:offset + take]
            offset += take
        return shards
    if partition == "dirichlet":
        return dirichlet_split_dataset(
            part_dataset,
            num_workers=num_workers,
            seed=seed,
            alpha=dirichlet_alpha,
            min_size=dirichlet_min_size,
        )

    rng = random.Random(seed)
    items = list(all_indices)
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


def training_nll_loss(log_probs, target, label_smoothing=0.0):
    if label_smoothing <= 0:
        return F.nll_loss(log_probs, target)
    nll = F.nll_loss(log_probs, target)
    smooth = -log_probs.mean(dim=1).mean()
    return (1.0 - label_smoothing) * nll + label_smoothing * smooth


def staleness_weight(tau, strategy, a, b):
    if strategy == "poly":
        return 1.0 / ((tau + 1) ** abs(a))
    if strategy == "hinge":
        return 1.0 / (a * (tau + b) + 1.0) if tau > b else 1.0
    return 1.0


def lr_schedule_scale(step, total_updates, args):
    if total_updates is None or total_updates <= 0:
        return 1.0
    if args.lr_schedule == "cosine":
        progress = min(1.0, float(step) / float(total_updates))
        min_ratio = args.lr_min_ratio
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))
    return 1.0


def clip_vector_norm(vec, max_norm, eps=1e-12):
    if max_norm is None or max_norm <= 0:
        return vec
    norm = torch.norm(vec)
    if norm <= max_norm:
        return vec
    return vec * (max_norm / (norm + eps))


def tree_tensor_map(obj, fn):
    if torch.is_tensor(obj):
        return fn(obj)
    if isinstance(obj, dict):
        return {key: tree_tensor_map(value, fn) for key, value in obj.items()}
    if isinstance(obj, list):
        return [tree_tensor_map(value, fn) for value in obj]
    if isinstance(obj, tuple):
        return tuple(tree_tensor_map(value, fn) for value in obj)
    return obj


def optimizer_state_to_cpu(state_dict):
    if state_dict is None:
        return None
    return tree_tensor_map(state_dict, lambda tensor: tensor.detach().cpu())


def optimizer_state_to_device(state_dict, device):
    if state_dict is None:
        return None
    return tree_tensor_map(state_dict, lambda tensor: tensor.detach().to(device))


def batch_loss_for_snapshot(model, snapshot, batch_cpu, device):
    data_cpu, target_cpu = batch_cpu
    set_flat_params(model, snapshot.to(next(model.parameters()).dtype))
    model.eval()
    with torch.no_grad():
        data = data_cpu.to(device)
        target = target_cpu.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)
    return float(loss.item())


class AsyncDistributedSAMServer:
    """Async center-side optimizer with AFL-style and SAM-style server updates."""

    def __init__(self, dim, device, dtype, args):
        self.alg = args.alg
        self.use_cv = args.alg.endswith("_cv") or args.alg == "fedac"
        self.lr = args.lr
        self.num_workers = args.num_workers
        self.total_updates = args.total_updates
        self.lr_schedule = args.lr_schedule
        self.lr_min_ratio = args.lr_min_ratio
        self.beta = args.sam_beta
        self.alpha = args.sam_alpha
        self.damp = args.sam_damp
        self.momentum = args.sam_momentum
        self.precond = args.sam_precond
        self.precond_beta = args.sam_precond_beta
        self.precond_eps = args.sam_precond_eps
        self.precond_init = args.sam_precond_init
        self.precond_min_denom = args.sam_precond_min_denom
        self.precond_max_scale = args.sam_precond_max_scale
        self.precond_warmup_updates = args.sam_precond_warmup_updates
        self.aa_warmup_updates = args.sam_aa_warmup_updates
        self.hist_length = args.sam_hist_length
        self.period = args.sam_period
        self.gamma = args.sam_gamma
        self.rtol = args.sam_rtol
        self.ridge = args.sam_ridge
        self.base_mix = args.sam_base_mix
        self.stale_base_mix = args.sam_stale_base_mix
        self.tau_base_mix = args.sam_tau_base_mix
        self.history_weight_exp = args.sam_history_weight_exp
        self.history_match_exp = args.sam_history_match_exp
        self.max_step_ratio = args.sam_max_step_ratio
        self.min_cosine = args.sam_min_cosine
        self.max_cond = args.sam_max_cond
        self.restart_on_reject = args.sam_restart_on_reject
        self.stop_updates = args.sam_stop_updates
        self.anchor_tol = args.sam_anchor_tol
        self.min_history = args.sam_min_history
        self.max_history_staleness = args.sam_max_history_staleness
        self.stale_strategy = args.stale_strategy
        self.stale_a = args.stale_a
        self.stale_b = args.stale_b
        self.eps = 1e-8
        self.global_control = torch.zeros(dim, device=device, dtype=dtype)
        self.counts = {
            "accepted": 0,
            "fallback": 0,
            "period_skip": 0,
            "insufficient_history": 0,
            "late_skip": 0,
            "anchor_reject": 0,
            "stale_skip": 0,
            "bad_conditioning": 0,
            "ratio_reject": 0,
            "cosine_reject": 0,
            "history_restart": 0,
            "warmup_skip": 0,
            "res_clip": 0,
            "xdelta_clip": 0,
            "buffer_wait": 0,
            "buffer_apply": 0,
        }

        self.step = 0
        self.aa_step = 0
        self.Xk = torch.zeros((dim, self.hist_length), device=device, dtype=dtype)
        self.Rk = torch.zeros((dim, self.hist_length), device=device, dtype=dtype)
        self.hist_taus = [None] * self.hist_length
        self.x_prev = torch.zeros(dim, device=device, dtype=dtype)
        self.res_prev = torch.zeros(dim, device=device, dtype=dtype)
        self.res_ema = torch.zeros(dim, device=device, dtype=dtype)
        self.res_ema_steps = 0
        self.grad_sq_ema = torch.full((dim,), self.precond_init, device=device, dtype=dtype)
        self.d_x_avg = torch.zeros(dim, device=device, dtype=dtype)
        self.d_res_avg = torch.zeros(dim, device=device, dtype=dtype)
        self.fedasync_decay = args.fedasync_decay
        self.fedbuff_k = args.fedbuff_k
        self.fedbuff_etag = args.fedbuff_etag
        self.fedbuff_buffer = []
        self.fadas_buffer_sum = torch.zeros(dim, device=device, dtype=dtype)
        self.fadas_buffer_count = 0
        self.fadas_max_delay = 0
        self.fadas_m = args.fadas_m
        self.fadas_tau_c = args.fadas_tau_c
        self.fadas_beta1 = args.fadas_beta1
        self.fadas_beta2 = args.fadas_beta2
        self.fadas_eps = args.fadas_eps
        self.fadas_eta = args.fadas_eta
        self.fadas_use_vhat = args.fadas_use_vhat == 1
        self.fadas_delay_adapt = args.fadas_delay_adapt == 1
        self.fadas_first_moment = torch.zeros(dim, device=device, dtype=dtype)
        self.fadas_second_moment = torch.zeros(dim, device=device, dtype=dtype)
        self.fadas_second_hat = torch.zeros(dim, device=device, dtype=dtype)
        self.ca2fl_eta = args.ca2fl_eta
        self.ca2fl_m = args.ca2fl_m
        self.ca2fl_buffer_count = 0
        self.ca2fl_calib_sum = torch.zeros(dim, device=device, dtype=dtype)
        self.ca2fl_h_cache = (
            [torch.zeros(dim, device=device, dtype=dtype) for _ in range(self.num_workers)]
            if self.alg == "ca2fl"
            else []
        )
        self.ca2fl_h_mean = torch.zeros(dim, device=device, dtype=dtype)
        self.ca2fl_round_h_mean = torch.zeros(dim, device=device, dtype=dtype)
        self.asyncaa_history = []
        self.fedac_beta1 = args.fedac_beta1
        self.fedac_beta2 = args.fedac_beta2
        self.fedac_eta_g = args.fedac_eta_g
        self.fedac_gamma = args.fedac_gamma
        self.fedac_buffer_size = args.fedac_buffer_size
        self.fedac_round = 0
        self.cv_server_lr = args.cv_server_lr
        self.cv_global_clip_norm = args.cv_global_clip_norm
        self.fedac_eps = 1e-8
        self.fedac_m = torch.zeros(dim, device=device, dtype=dtype)
        self.fedac_v = torch.zeros(dim, device=device, dtype=dtype)
        self.fedac_delta_buffer = []
        self.fedac_control_buffer = []
        self.fedac_snapshot_buffer = []

    def current_lr(self):
        return self.lr * lr_schedule_scale(self.step, self.total_updates, self)

    def current_fedac_local_lr(self):
        # Match AFL-Lib AsyncBaseClient.reset_optimizer():
        # lr * gamma^(server_round / concurrency). In this scaffold all
        # workers stay active, so concurrency equals num_workers.
        concurrency = max(1, self.num_workers)
        return self.lr * (self.fedac_gamma ** (float(self.fedac_round) / float(concurrency)))

    def _valid_history_indices(self):
        valid = []
        for idx, tau in enumerate(self.hist_taus):
            if tau is None:
                continue
            if self.max_history_staleness is not None and tau > self.max_history_staleness:
                continue
            valid.append(idx)
        return valid

    def _restart_history(self, xk, res):
        self.Xk.zero_()
        self.Rk.zero_()
        self.hist_taus = [None] * self.hist_length
        self.d_x_avg.zero_()
        self.d_res_avg.zero_()
        self.x_prev.copy_(xk)
        self.res_prev.copy_(res)
        self.aa_step = 1
        self.counts["history_restart"] += 1

    def _solve_anderson_coefficients(self, residual_matrix):
        gram = residual_matrix.t() @ residual_matrix
        if self.ridge > 0:
            gram = gram + self.ridge * torch.eye(
                gram.shape[0], device=gram.device, dtype=gram.dtype
            )
        ones = torch.ones(gram.shape[0], device=gram.device, dtype=gram.dtype)
        if gram.device.type == "npu":
            inv_ones = torch.linalg.pinv(gram.cpu(), hermitian=True, rtol=self.rtol) @ ones.cpu()
            inv_ones = inv_ones.to(gram.device)
        else:
            inv_ones = torch.linalg.pinv(gram, hermitian=True, rtol=self.rtol) @ ones
        denom = ones.dot(inv_ones)
        if abs(float(denom.item())) <= self.eps:
            raise RuntimeError("degenerate Anderson system")
        return inv_ones / denom

    def _history_gram(self, matrix):
        """Compute M^T M via pairwise dots to avoid large GEMM workspaces."""
        width = matrix.shape[1]
        gram = torch.empty((width, width), device=matrix.device, dtype=matrix.dtype)
        for i in range(width):
            col_i = matrix[:, i]
            diag_val = col_i.dot(col_i)
            gram[i, i] = diag_val
            for j in range(i + 1, width):
                val = col_i.dot(matrix[:, j])
                gram[i, j] = val
                gram[j, i] = val
        return gram

    def _history_rhs(self, matrix, vector):
        """Compute M^T v via columnwise dots for memory-heavy models."""
        width = matrix.shape[1]
        rhs = torch.empty(width, device=matrix.device, dtype=matrix.dtype)
        for i in range(width):
            rhs[i] = matrix[:, i].dot(vector)
        return rhs

    def _combine_sam_history(self, res, X_hist, R_hist, gamma_k):
        """Form beta * r - (alpha X + alpha beta R) gamma without tall GEMM."""
        x_delta = self.beta * res.clone()
        for i in range(gamma_k.shape[0]):
            coeff = float(gamma_k[i].item())
            if abs(coeff) <= self.eps:
                continue
            x_delta.add_(X_hist[:, i], alpha=-(self.alpha * coeff))
            x_delta.add_(R_hist[:, i], alpha=-(self.alpha * self.beta * coeff))
        return x_delta

    def _compute_preconditioned_residual(self, base_res, grad, weight):
        res = base_res
        effective_lr = self.current_lr()
        if self.precond == "rms":
            self.grad_sq_ema.mul_(self.precond_beta).addcmul_(grad, grad, value=1.0 - self.precond_beta)
            denom = self.grad_sq_ema.sqrt().add(self.precond_eps)
            if self.precond_min_denom > 0:
                denom = torch.clamp_min(denom, self.precond_min_denom)
            res = (-effective_lr * weight) * (grad / denom)
        elif self.precond == "adagrad":
            self.grad_sq_ema.addcmul_(grad, grad, value=1.0)
            denom = self.grad_sq_ema.sqrt().add(self.precond_eps)
            if self.precond_min_denom > 0:
                denom = torch.clamp_min(denom, self.precond_min_denom)
            res = (-effective_lr * weight) * (grad / denom)

        if self.precond != "none" and self.precond_warmup_updates > 0:
            warm = min(1.0, float(self.step + 1) / float(self.precond_warmup_updates))
            res = (1.0 - warm) * base_res + warm * res
        if self.precond != "none" and self.precond_max_scale > 0:
            base_norm = torch.norm(base_res)
            res_norm = torch.norm(res)
            max_norm = self.precond_max_scale * base_norm
            if res_norm > max_norm and max_norm > self.eps:
                res = res * (max_norm / (res_norm + self.eps))
                self.counts["res_clip"] += 1
        if self.momentum > 0:
            self.res_ema.mul_(self.momentum).add_(res, alpha=1.0 - self.momentum)
            self.res_ema_steps += 1
            bias_correction = 1.0 - (self.momentum ** self.res_ema_steps)
            res = self.res_ema / max(bias_correction, self.eps)
        return res

    def _asyncaa_update(self, xk, res, tau):
        # GD-style AsyncAA:
        # x_k = current server iterate
        # y_k = x_k + r_k is the base async descent iterate
        # r_k is the async residual / base step
        y_base = xk + res
        current_too_stale = (
            self.max_history_staleness is not None and tau > self.max_history_staleness
        )
        if not current_too_stale:
            self.asyncaa_history.append({
                "y": y_base.detach().clone(),
                "r": res.detach().clone(),
                "tau": tau,
            })
            if len(self.asyncaa_history) > self.hist_length:
                self.asyncaa_history = self.asyncaa_history[-self.hist_length:]
        else:
            self.counts["stale_skip"] += 1

        window = min(self.fedbuff_k, len(self.asyncaa_history))
        if window < max(2, self.min_history):
            self.counts["insufficient_history"] += 1
            self.step += 1
            return y_base, y_base

        if self.period > 1 and self.step > 0 and (self.step % self.period) != 0:
            self.counts["period_skip"] += 1
            self.step += 1
            return y_base, y_base

        history = self.asyncaa_history[-window:]
        residual_matrix = torch.stack([item["r"] for item in history], dim=1)
        y_matrix = torch.stack([item["y"] for item in history], dim=1)
        if self.history_match_exp > 0:
            tau_hist = torch.tensor(
                [item["tau"] for item in history],
                device=residual_matrix.device,
                dtype=residual_matrix.dtype,
            )
            match_weights = 1.0 / ((tau_hist - float(tau)).abs() + 1.0) ** self.history_match_exp
            residual_matrix = residual_matrix * match_weights.unsqueeze(0)
            y_matrix = y_matrix * match_weights.unsqueeze(0)

        cond_input = residual_matrix.cpu() if residual_matrix.device.type == "npu" else residual_matrix
        cond_value = torch.linalg.cond(cond_input).item()
        if (not math.isfinite(cond_value)) or (self.max_cond > 0 and cond_value > self.max_cond):
            self.counts["bad_conditioning"] += 1
            self.asyncaa_history = self.asyncaa_history[-1:]
            self.step += 1
            return y_base, y_base

        try:
            alpha = self._solve_anderson_coefficients(residual_matrix)
        except RuntimeError:
            self.counts["fallback"] += 1
            self.asyncaa_history = self.asyncaa_history[-1:]
            self.step += 1
            return y_base, y_base

        alpha = alpha * self.beta
        alpha[-1] += 1.0 - self.beta
        x_candidate = y_matrix @ alpha
        if not torch.isfinite(x_candidate).all():
            self.counts["fallback"] += 1
            self.asyncaa_history = self.asyncaa_history[-1:]
            self.step += 1
            return y_base, y_base

        # Approximate candidate residual by the affine combination of the
        # recent residual history; this keeps acceptance iterate-based.
        mixed_residual = residual_matrix @ alpha
        base_norm = torch.norm(history[-1]["r"]).item()
        cand_norm = torch.norm(mixed_residual).item()
        if self.max_step_ratio > 0 and cand_norm > self.max_step_ratio * max(base_norm, self.eps):
            self.counts["ratio_reject"] += 1
            self.step += 1
            return y_base, y_base
        if self.anchor_tol > 0:
            anchor_gap = torch.norm(x_candidate - y_base).item()
            if anchor_gap > self.anchor_tol * max(base_norm, self.eps):
                self.counts["anchor_reject"] += 1
                self.step += 1
                return y_base, y_base

        mix = self.base_mix
        if current_too_stale:
            mix = max(mix, self.stale_base_mix)
        if self.tau_base_mix > 0:
            if self.max_history_staleness is not None and self.max_history_staleness > 0:
                tau_ratio = min(1.0, float(tau) / float(self.max_history_staleness))
            else:
                tau_ratio = float(tau) / (float(tau) + 1.0)
            mix = max(mix, min(1.0, self.tau_base_mix * tau_ratio))

        if mix > 0:
            x_candidate = (1.0 - mix) * x_candidate + mix * y_base

        self.counts["accepted"] += 1
        self.counts["buffer_apply"] += 1
        self.step += 1
        return x_candidate, y_base

    def update(self, xk, worker_grad, tau, snapshot=None, delta_control=None, worker_delta=None, worker_id=None):
        weight = staleness_weight(tau, self.stale_strategy, self.stale_a, self.stale_b)
        effective_lr = self.current_lr()

        if self.alg == "fedac":
            self.fedac_round += 1
            # FedAC consumes uploaded client deltas dW and control deltas dC
            # produced by the worker after local SGD and control correction.
            if worker_delta is not None:
                dW = worker_delta.to(xk.dtype)
            else:
                grad = worker_grad.to(xk.dtype)
                dW = (-effective_lr) * grad
            self.fedac_delta_buffer.append(dW.detach().clone())
            self.fedac_control_buffer.append(
                delta_control.detach().clone() if delta_control is not None else torch.zeros_like(dW)
            )
            self.fedac_snapshot_buffer.append(snapshot.detach().clone() if snapshot is not None else xk.detach().clone())
            if len(self.fedac_delta_buffer) < self.fedac_buffer_size:
                self.counts["buffer_wait"] += 1
                return xk, xk

            r_list = []
            for delta_i, snap_i in zip(self.fedac_delta_buffer, self.fedac_snapshot_buffer):
                delta_global = (xk - snap_i) if self.fedac_round > self.fedac_buffer_size else xk
                if torch.norm(delta_global) <= self.eps or torch.norm(delta_i) <= self.eps:
                    r_val = 0.1
                else:
                    r_val = F.cosine_similarity(delta_global, delta_i, dim=0).item()
                    r_val = max(r_val, 0.1)
                r_list.append(r_val)
            weight_sum = sum(r_list) + self.fedac_eps
            w_list = [r / weight_sum for r in r_list]

            dW_mix = sum(w * dW_i for w, dW_i in zip(w_list, self.fedac_delta_buffer))
            self.fedac_m = self.fedac_beta1 * self.fedac_m + (1.0 - self.fedac_beta1) * dW_mix
            self.fedac_v = self.fedac_beta2 * self.fedac_v + (1.0 - self.fedac_beta2) * (dW_mix ** 2)
            m_hat = self.fedac_beta1 * self.fedac_m + (1.0 - self.fedac_beta1) * dW_mix
            x_next = xk + self.fedac_eta_g * m_hat / (torch.sqrt(self.fedac_v) + self.fedac_eps)
            delta_control_mix = sum(w * dC_i for w, dC_i in zip(w_list, self.fedac_control_buffer))
            self.global_control.add_(delta_control_mix)

            self.fedac_delta_buffer.clear()
            self.fedac_control_buffer.clear()
            self.fedac_snapshot_buffer.clear()
            self.counts["buffer_apply"] += 1
            self.step += 1
            return x_next, x_next

        if self.alg == "fadas":
            # FADAS consumes client model update differences as pseudo-gradients:
            # dW = local_model_after - local_model_before. In this minibatch
            # simulator, the corresponding local model delta is -lr * grad.
            grad = worker_grad.to(xk.dtype)
            dW = worker_delta.to(xk.dtype) if worker_delta is not None else (-effective_lr) * grad
            self.fadas_buffer_sum.add_(dW.detach())
            self.fadas_buffer_count += 1
            self.fadas_max_delay = max(self.fadas_max_delay, int(tau))
            if self.fadas_buffer_count < self.fadas_m:
                self.counts["buffer_wait"] += 1
                return xk, xk

            delta_t = self.fadas_buffer_sum / float(self.fadas_buffer_count)
            self.fadas_first_moment.mul_(self.fadas_beta1).add_(delta_t, alpha=1.0 - self.fadas_beta1)
            self.fadas_second_moment.mul_(self.fadas_beta2).addcmul_(
                delta_t, delta_t, value=1.0 - self.fadas_beta2
            )
            self.fadas_second_hat.copy_(torch.maximum(self.fadas_second_hat, self.fadas_second_moment))
            denom_source = self.fadas_second_hat if self.fadas_use_vhat else self.fadas_second_moment
            eta_t = self.fadas_eta
            if self.fadas_delay_adapt and self.fadas_max_delay > self.fadas_tau_c:
                # The paper describes delay-adaptive scaling proportional to
                # 1 / tau_max; AFL-Lib also uses this practical eta/tau form.
                eta_t = self.fadas_eta / float(max(1, self.fadas_max_delay))
            x_next = xk + eta_t * self.fadas_first_moment / (torch.sqrt(denom_source) + self.fadas_eps)
            if not torch.isfinite(x_next).all():
                self.counts["fallback"] += 1
                x_next = xk

            self.fadas_buffer_sum.zero_()
            self.fadas_buffer_count = 0
            self.fadas_max_delay = 0
            self.counts["buffer_apply"] += 1
            self.step += 1
            return x_next, x_next

        if self.alg == "ca2fl":
            if worker_id is None:
                raise RuntimeError("CA2FL requires worker_id for per-client cached updates")
            # CA2FL maintains each client's latest update h_i and forms
            # v_t = h_t + mean_i(dW_i - h_i), where h_t is the round-start
            # average cache. This fixes the AFL-Lib bookkeeping issue where
            # buffer_clients is never recorded.
            grad = worker_grad.to(xk.dtype)
            dW = worker_delta.to(xk.dtype) if worker_delta is not None else (-effective_lr) * grad
            old_cache = self.ca2fl_h_cache[worker_id]
            calibrated_delta = dW - old_cache
            self.ca2fl_calib_sum.add_(calibrated_delta.detach())
            self.ca2fl_buffer_count += 1
            self.ca2fl_h_mean.add_(calibrated_delta.detach(), alpha=1.0 / float(self.num_workers))
            old_cache.copy_(dW.detach())
            if self.ca2fl_buffer_count < self.ca2fl_m:
                self.counts["buffer_wait"] += 1
                return xk, xk

            v_t = self.ca2fl_round_h_mean + self.ca2fl_calib_sum / float(self.ca2fl_buffer_count)
            x_next = xk + self.ca2fl_eta * v_t
            if not torch.isfinite(x_next).all():
                self.counts["fallback"] += 1
                x_next = xk

            self.ca2fl_round_h_mean.copy_(self.ca2fl_h_mean)
            self.ca2fl_calib_sum.zero_()
            self.ca2fl_buffer_count = 0
            self.counts["buffer_apply"] += 1
            self.step += 1
            return x_next, x_next

        grad = worker_grad.to(xk.dtype)
        base_res = -effective_lr * weight * grad
        if self.alg == "fedasync":
            x_next = xk + self.fedasync_decay * base_res
            self.step += 1
            return x_next, x_next

        if self.alg == "fedbuff":
            # FedBuff still buffers recent base async steps; when a
            # preconditioner is enabled, the buffered step becomes the
            # preconditioned base step instead of the raw SGD step.
            fedbuff_res = self._compute_preconditioned_residual(base_res, grad, weight)
            self.fedbuff_buffer.append(fedbuff_res.detach().clone())
            if len(self.fedbuff_buffer) < self.fedbuff_k:
                self.counts["buffer_wait"] += 1
                return xk, xk
            stacked = torch.stack(self.fedbuff_buffer, dim=0)
            # Paper-style FedBuff applies the server step to the average of the
            # K buffered updates. The averaging already contributes the 1/K
            # factor, so we should not divide by K a second time here.
            x_next = xk + self.fedbuff_etag * stacked.mean(dim=0)
            self.fedbuff_buffer.clear()
            self.counts["buffer_apply"] += 1
            self.step += 1
            return x_next, x_next

        # r_k is the async center descent step from the stale worker gradient.
        res = self._compute_preconditioned_residual(base_res, grad, weight)

        if self.alg == "asyncaa":
            return self._asyncaa_update(xk, res, tau)

        if self.alg in ("asyncsgd", "asyncsgd_cv"):
            self.x_prev.copy_(xk)
            self.res_prev.copy_(res)
            self.step += 1
            return xk + res, xk + res

        if self.step < self.aa_warmup_updates:
            self.x_prev.copy_(xk)
            self.res_prev.copy_(res)
            self.counts["warmup_skip"] += 1
            self.step += 1
            return xk + res, xk + res

        cnt = self.step
        if self.stop_updates is not None and cnt >= self.stop_updates:
            self.x_prev.copy_(xk)
            self.res_prev.copy_(res)
            self.counts["late_skip"] += 1
            self.step += 1
            return xk + res, xk + res

        current_too_stale = (
            self.max_history_staleness is not None and tau > self.max_history_staleness
        )
        x_next = xk + res

        has_history = self.aa_step > 0
        hist_weight = weight ** self.history_weight_exp if self.history_weight_exp > 0 else 1.0
        delta_x_ref = None
        if has_history and not current_too_stale:
            self.d_x_avg.mul_(self.gamma).add_(xk - self.x_prev, alpha=1 - self.gamma)
            self.d_res_avg.mul_(self.gamma).add_(res - self.res_prev, alpha=1 - self.gamma)
            k = (self.aa_step - 1) % self.hist_length
            self.Xk[:, k] = hist_weight * self.d_x_avg
            self.Rk[:, k] = hist_weight * self.d_res_avg
            self.hist_taus[k] = tau
            delta_x_ref = self.Xk[:, k]

        if not current_too_stale:
            self.x_prev.copy_(xk)
            self.res_prev.copy_(res)
            self.aa_step += 1
        else:
            # Over-stale arrivals do not pollute AA memory, but they can still be
            # blended with an AA candidate built from the fresh residual history.
            self.counts["stale_skip"] += 1

        if not has_history:
            self.counts["insufficient_history"] += 1
            self.step += 1
            return x_next, xk + res

        if self.aa_step % self.period != 0:
            self.counts["period_skip"] += 1
            self.step += 1
            return x_next, xk + res

        valid_indices = self._valid_history_indices()
        if len(valid_indices) < self.min_history:
            self.counts["insufficient_history"] += 1
            self.step += 1
            return x_next, xk + res

        X_hist = self.Xk[:, valid_indices]
        R_hist = self.Rk[:, valid_indices]
        if self.history_match_exp > 0:
            tau_hist = torch.tensor(
                [self.hist_taus[idx] for idx in valid_indices],
                device=X_hist.device,
                dtype=X_hist.dtype,
            )
            match_weights = 1.0 / ((tau_hist - float(tau)).abs() + 1.0) ** self.history_match_exp
            X_hist = X_hist * match_weights.unsqueeze(0)
            R_hist = R_hist * match_weights.unsqueeze(0)
        if delta_x_ref is None:
            delta_x_ref = X_hist[:, -1]

        delta = self.damp * res.dot(res) / (delta_x_ref.dot(delta_x_ref) + self.eps)
        # For wide models such as CIFAR-style ResNeXt, explicit tall GEMMs can
        # request a large temporary workspace on NPU. These pairwise-dot
        # contractions are algebraically equivalent but much more memory stable.
        system = self._history_gram(R_hist) + delta * self._history_gram(X_hist)
        if self.ridge > 0:
            system = system + self.ridge * torch.eye(system.shape[0], device=system.device, dtype=system.dtype)
        if self.max_cond > 0:
            cond_value = torch.linalg.cond(system.cpu() if system.device.type == "npu" else system).item()
            if (not math.isfinite(cond_value)) or cond_value > self.max_cond:
                self.counts["bad_conditioning"] += 1
                if self.restart_on_reject:
                    self._restart_history(xk, res)
                self.step += 1
                return x_next, xk + res
        rhs = self._history_rhs(R_hist, res)
        if system.device.type == "npu":
            gamma_k = torch.linalg.pinv(system.cpu(), hermitian=True, rtol=self.rtol) @ rhs.cpu()
            gamma_k = gamma_k.to(system.device)
        else:
            gamma_k = torch.linalg.pinv(system, hermitian=True, rtol=self.rtol) @ rhs
        x_delta = self._combine_sam_history(res, X_hist, R_hist, gamma_k)

        res_norm = torch.norm(res).item()
        delta_norm = torch.norm(x_delta).item()
        if self.max_step_ratio > 0 and res_norm > self.eps and delta_norm > self.max_step_ratio * res_norm:
            x_delta = x_delta * ((self.max_step_ratio * res_norm) / (delta_norm + self.eps))
            delta_norm = torch.norm(x_delta).item()
            self.counts["xdelta_clip"] += 1
        if self.max_step_ratio > 0 and res_norm > self.eps and delta_norm > self.max_step_ratio * res_norm:
            self.counts["ratio_reject"] += 1
            if self.restart_on_reject:
                self._restart_history(xk, res)
            self.step += 1
            return x_next, xk + res
        if self.min_cosine > 0 and res_norm > self.eps and delta_norm > self.eps:
            cosine = float(x_delta.dot(res).item()) / (delta_norm * res_norm + self.eps)
            if cosine < self.min_cosine:
                self.counts["cosine_reject"] += 1
                if self.restart_on_reject:
                    self._restart_history(xk, res)
                self.step += 1
                return x_next, xk + res

        if self.anchor_tol > 0:
            if res_norm > self.eps and delta_norm > self.anchor_tol * res_norm:
                self.counts["anchor_reject"] += 1
                if self.restart_on_reject:
                    self._restart_history(xk, res)
                self.step += 1
                return x_next, xk + res

        if torch.isfinite(x_delta).all() and x_delta.dot(res) > 0:
            x_aa = xk + x_delta
            mix = self.base_mix
            if self.tau_base_mix > 0:
                if self.max_history_staleness is not None and self.max_history_staleness > 0:
                    tau_ratio = min(1.0, float(tau) / float(self.max_history_staleness))
                else:
                    tau_ratio = float(tau) / (float(tau) + 1.0)
                mix = max(mix, min(1.0, self.tau_base_mix * tau_ratio))
            if current_too_stale:
                mix = max(mix, self.stale_base_mix)
            if mix > 0:
                x_next = (1.0 - mix) * x_aa + mix * (xk + res)
            else:
                x_next = x_aa
            self.counts["accepted"] += 1
        else:
            self.counts["fallback"] += 1

        self.step += 1
        return x_next, xk + res


def worker_delay(worker_id, dispatch_step, args):
    # Deterministic heterogeneity so async arrival order is reproducible.
    tier = worker_id % 3
    base = 1.0 + tier * args.delay_gap
    jitter_seed = f"{args.seed}:{worker_id}:{dispatch_step}"
    rng = random.Random(jitter_seed)
    jitter = args.delay_jitter * rng.random()
    return base + jitter


def next_batch(loader, loader_iter):
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter), loader_iter


def compute_worker_grad(
    worker_model,
    snapshot,
    batch,
    device,
    dtype,
    local_control=None,
    global_control=None,
    grad_clip_norm=0.0,
    cv_momentum=1.0,
    cv_delta_clip_norm=0.0,
    weight_decay=0.0,
    label_smoothing=0.0,
):
    set_flat_params(worker_model, snapshot.to(next(worker_model.parameters()).dtype))
    data, target = batch
    batch_cpu = (data.clone(), target.clone())
    data, target = data.to(device), target.to(device)
    worker_model.zero_grad()
    output = worker_model(data)
    loss = training_nll_loss(output, target, label_smoothing)
    loss.backward()
    raw_grad = flat_grads(worker_model).to(dtype)
    if weight_decay > 0:
        raw_grad = raw_grad + weight_decay * flat_params(worker_model).to(dtype)
    raw_grad = clip_vector_norm(raw_grad, grad_clip_norm)
    if local_control is not None and global_control is not None:
        # SCAFFOLD-style corrected gradient on the worker:
        # g_i^cv = g_i - c_i + c.
        grad = raw_grad - local_control + global_control
        delta_control = cv_momentum * (raw_grad - local_control)
        delta_control = clip_vector_norm(delta_control, cv_delta_clip_norm)
        new_local_control = (local_control + delta_control).detach().clone()
    else:
        grad = raw_grad
        new_local_control = None
        delta_control = None
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return {
        "grad": grad.detach().clone(),
        "raw_grad": raw_grad.detach().clone(),
        "new_local_control": new_local_control,
        "delta_control": delta_control,
        "loss": float(loss.item()),
        "correct": correct,
        "size": len(data),
        "batch_cpu": batch_cpu,
    }


def compute_fedac_worker_payload(
    worker_model,
    snapshot,
    loader,
    device,
    dtype,
    local_control,
    global_control,
    local_lr,
    local_epochs=5,
    local_momentum=0.9,
    weight_decay=0.0,
    optimizer_state=None,
    label_smoothing=0.0,
):
    # Exact FedAC client semantics, specialized to the current async scaffold:
    # start from the downloaded global snapshot, run local SGD over the full
    # worker loader for E local epochs, then apply the control correction to
    # form dW and dC exactly as in AFL-Lib/fedac.py.
    set_flat_params(worker_model, snapshot.to(next(worker_model.parameters()).dtype))
    worker_model.train()
    prev_model = flat_params(worker_model).to(dtype).detach().clone()
    if local_control is None:
        local_control = torch.zeros_like(prev_model)
    else:
        local_control = local_control.to(dtype)
    if global_control is None:
        global_control = torch.zeros_like(prev_model)
    else:
        global_control = global_control.to(dtype)

    total_loss = 0.0
    total_correct = 0
    total_size = 0
    last_batch_cpu = None
    last_raw_grad = torch.zeros_like(prev_model)
    optim = torch.optim.SGD(
        worker_model.parameters(),
        lr=local_lr,
        momentum=local_momentum,
        weight_decay=weight_decay,
    )
    if optimizer_state is not None:
        optim.load_state_dict(optimizer_state_to_device(optimizer_state, device))
    for param_group in optim.param_groups:
        param_group["lr"] = local_lr
        param_group["momentum"] = local_momentum
        param_group["weight_decay"] = weight_decay

    for _ in range(max(1, local_epochs)):
        for data, target in loader:
            batch_cpu = (data.clone(), target.clone())
            data, target = data.to(device), target.to(device)
            optim.zero_grad()
            output = worker_model(data)
            loss = training_nll_loss(output, target, label_smoothing)
            loss.backward()
            raw_grad = flat_grads(worker_model).to(dtype)
            optim.step()

            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            total_size += len(data)
            total_loss += float(loss.item()) * len(data)
            last_batch_cpu = batch_cpu
            last_raw_grad = raw_grad.detach().clone()

    cur_model = flat_params(worker_model).to(dtype).detach().clone()
    control_gap = global_control - local_control
    corrected_model = cur_model - local_lr * control_gap
    set_flat_params(worker_model, corrected_model.to(next(worker_model.parameters()).dtype))

    # Original FedAC client notation:
    # prev_model = downloaded model before local train
    # corrected_model = local model after SGD steps and the control correction
    # dW = corrected_model - prev_model
    # hat_C = (prev_model - corrected_model) / (E * lr) - (C - C_i)
    # dC = hat_C - C_i
    denom = max(1, local_epochs) * max(float(local_lr), 1e-12)
    hat_control = (prev_model - corrected_model) / denom - control_gap
    delta_control = hat_control - local_control

    avg_loss = total_loss / max(1, total_size)
    return {
        "grad": last_raw_grad.detach().clone(),
        "raw_grad": last_raw_grad.detach().clone(),
        "worker_delta": (corrected_model - prev_model).detach().clone(),
        "new_local_control": hat_control.detach().clone(),
        "delta_control": delta_control.detach().clone(),
        "optimizer_state": optimizer_state_to_cpu(optim.state_dict()),
        "loss": float(avg_loss),
        "correct": total_correct,
        "size": total_size,
        "batch_cpu": last_batch_cpu,
        "snapshot": prev_model.detach().clone(),
    }


def main():
    parser = argparse.ArgumentParser(description="Async distributed experiment for SAM")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10", "cifar100"])
    parser.add_argument(
        "--model",
        type=str,
        default="auto",
        choices=[
            "auto",
            "cnn",
            "resnet18",
            "resnet20",
            "resnet32",
            "resnet44",
            "resnet56",
            "resnet110",
            "resnet1202",
            "resnext29_4x24d",
            "resnext29_8x16d",
            "resnext29_16x8d",
            "resnext29_8x64d",
        ],
    )
    parser.add_argument("--train-part-size", type=int, default=12000)
    parser.add_argument("--test-part-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=600)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr-schedule", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--lr-min-ratio", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--cifar-augment",
        type=str,
        default="basic",
        choices=["none", "basic", "randaugment", "autoaugment", "trivial"],
    )
    parser.add_argument("--random-erasing", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--partition", type=str, default="iid", choices=["iid", "round_robin", "label_sorted", "dirichlet"])
    parser.add_argument("--dirichlet-alpha", type=float, default=0.05)
    parser.add_argument("--dirichlet-min-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "npu"])
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--dump-data", default="async_distributed_output.pkl", type=str)
    parser.add_argument(
        "--alg",
        type=str,
        default="asyncsam",
        choices=[
            "asyncsgd",
            "asyncsam",
            "asyncsgd_cv",
            "asyncsam_cv",
            "fedasync",
            "fedbuff",
            "fedac",
            "asyncaa",
            "fadas",
            "ca2fl",
        ],
    )
    parser.add_argument("--fedasync-decay", type=float, default=1.0)
    parser.add_argument("--fedbuff-k", type=int, default=3)
    parser.add_argument("--fedbuff-etag", type=float, default=5.0)
    parser.add_argument("--fadas-m", type=int, default=5)
    parser.add_argument("--fadas-tau-c", type=int, default=1)
    parser.add_argument("--fadas-beta1", type=float, default=0.9)
    parser.add_argument("--fadas-beta2", type=float, default=0.99)
    parser.add_argument("--fadas-eps", type=float, default=1e-8)
    parser.add_argument("--fadas-eta", type=float, default=0.001)
    parser.add_argument("--fadas-use-vhat", type=int, default=1, choices=[0, 1])
    parser.add_argument("--fadas-delay-adapt", type=int, default=1, choices=[0, 1])
    parser.add_argument("--ca2fl-m", type=int, default=10)
    parser.add_argument("--ca2fl-eta", type=float, default=0.01)
    parser.add_argument("--fedac-beta1", type=float, default=0.6)
    parser.add_argument("--fedac-beta2", type=float, default=0.9)
    # We keep the tuned legacy migration path as the default FedAC variant for
    # shared comparisons because it consistently outperformed the stricter
    # AFL-Lib-style exact path in this codebase. The exact variant remains
    # available for ablations via --fedac-client-mode exact.
    parser.add_argument("--fedac-client-mode", type=str, default="legacy", choices=["legacy", "exact"])
    parser.add_argument("--fedac-buffer-size", type=int, default=5)
    parser.add_argument("--fedac-eta-g", type=float, default=0.001)
    parser.add_argument("--fedac-gamma", type=float, default=0.99)
    parser.add_argument("--fedac-local-epochs", type=int, default=5)
    parser.add_argument("--fedac-local-momentum", type=float, default=0.9)
    parser.add_argument("--fedac-local-weight-decay", type=float, default=1e-4)
    parser.add_argument("--fedac-lr-mode", type=str, default="shared", choices=["afl", "shared"])
    parser.add_argument("--fedac-persist-optimizer", type=int, default=0, choices=[0, 1])
    # Deprecated legacy knob from earlier surrogate FedAC experiments; the
    # exact FedAC path now follows local epochs instead.
    parser.add_argument("--fedac-local-steps", type=int, default=1)
    parser.add_argument("--sam-period", type=int, default=1)
    parser.add_argument("--sam-hist-length", type=int, default=10)
    parser.add_argument("--sam-alpha", type=float, default=1.0)
    parser.add_argument("--sam-beta", type=float, default=1.0)
    parser.add_argument("--sam-damp", type=float, default=1e-2)
    parser.add_argument("--sam-gamma", type=float, default=0.9)
    parser.add_argument("--sam-momentum", type=float, default=0.0)
    parser.add_argument("--sam-precond", type=str, default="none", choices=["none", "rms", "adagrad"])
    parser.add_argument("--sam-precond-beta", type=float, default=0.99)
    parser.add_argument("--sam-precond-eps", type=float, default=1e-8)
    parser.add_argument("--sam-precond-init", type=float, default=1.0)
    parser.add_argument("--sam-precond-min-denom", type=float, default=0.0)
    parser.add_argument("--sam-precond-max-scale", type=float, default=0.0)
    parser.add_argument("--sam-precond-warmup-updates", type=int, default=0)
    parser.add_argument("--sam-aa-warmup-updates", type=int, default=0)
    parser.add_argument("--sam-rtol", type=float, default=1e-4)
    parser.add_argument("--sam-ridge", type=float, default=0.0)
    parser.add_argument("--sam-base-mix", type=float, default=0.0)
    parser.add_argument("--sam-stale-base-mix", type=float, default=0.0)
    parser.add_argument("--sam-tau-base-mix", type=float, default=0.0)
    parser.add_argument("--sam-history-weight-exp", type=float, default=0.0)
    parser.add_argument("--sam-history-match-exp", type=float, default=0.0)
    parser.add_argument("--sam-max-step-ratio", type=float, default=0.0)
    parser.add_argument("--sam-min-cosine", type=float, default=0.0)
    parser.add_argument("--sam-max-cond", type=float, default=0.0)
    parser.add_argument("--sam-restart-on-reject", action="store_true", default=False)
    parser.add_argument("--sam-batch-accept", action="store_true", default=False)
    parser.add_argument("--sam-batch-tol", type=float, default=0.0)
    parser.add_argument("--sam-min-history", type=int, default=1)
    parser.add_argument("--sam-max-history-staleness", type=int, default=-1)
    parser.add_argument("--sam-anchor-tol", type=float, default=0.0)
    parser.add_argument("--sam-stop-fraction", type=float, default=1.0)
    parser.add_argument("--cv-server-lr", type=float, default=1.0)
    parser.add_argument("--cv-momentum", type=float, default=1.0)
    parser.add_argument("--cv-delta-clip-norm", type=float, default=0.0)
    parser.add_argument("--cv-global-clip-norm", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--partial-dump-every-epoch", type=int, default=1)
    parser.add_argument("--early-abort-epoch", type=int, default=0)
    parser.add_argument("--early-abort-min-acc", type=float, default=-1.0)
    parser.add_argument("--early-abort-max-loss", type=float, default=0.0)
    parser.add_argument("--precision", type=int, default=1, choices=[0, 1])
    parser.add_argument("--stale-strategy", type=str, default="hinge", choices=["constant", "poly", "hinge"])
    parser.add_argument("--stale-a", type=float, default=1.0)
    parser.add_argument("--stale-b", type=float, default=4.0)
    parser.add_argument("--delay-gap", type=float, default=0.5)
    parser.add_argument("--delay-jitter", type=float, default=0.1)
    args = parser.parse_args()

    if args.no_cuda and args.device == "auto":
        args.device = "cpu"
    device = resolve_device(args.device)
    set_seed(args.seed, device.type)

    dataset_train, dataset_test = build_dataset(
        args.dataset,
        cifar_augment=args.cifar_augment,
        random_erasing=args.random_erasing,
    )
    train_subset = make_random_subset(dataset_train, args.train_part_size, args.seed)
    test_subset = make_random_subset(dataset_test, args.test_part_size, args.seed + 1)

    worker_shards = split_dataset(
        train_subset,
        args.num_workers,
        args.partition,
        args.seed,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_min_size=args.dirichlet_min_size,
    )
    worker_loaders = [
        torch.utils.data.DataLoader(Subset(train_subset, shard), batch_size=args.batch_size, shuffle=True)
        for shard in worker_shards
    ]
    worker_iters = [iter(loader) for loader in worker_loaders]
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=args.test_batch_size, shuffle=False)

    if args.alg == "fedac":
        # One async FedAC event corresponds to one worker completing a full
        # local-training run, so count epochs in worker-events instead of
        # mini-batches.
        updates_per_epoch = args.num_workers
    else:
        updates_per_epoch = sum(len(loader) for loader in worker_loaders)
    total_updates = args.epochs * updates_per_epoch
    args.total_updates = total_updates
    if args.sam_max_history_staleness < 0:
        args.sam_max_history_staleness = None
    args.sam_stop_updates = None
    if args.alg in ("asyncsam", "asyncsam_cv") and args.sam_stop_fraction < 1.0:
        args.sam_stop_updates = max(1, int(total_updates * args.sam_stop_fraction))

    server_model = build_model(args.dataset, args.model).to(device)
    worker_model = build_model(args.dataset, args.model).to(device)
    if device.type == "npu":
        dtype = next(server_model.parameters()).dtype
    else:
        dtype = torch.float64 if args.precision == 1 else next(server_model.parameters()).dtype
    dim = flat_params(server_model).numel()
    server = AsyncDistributedSAMServer(dim=dim, device=device, dtype=dtype, args=args)
    worker_controls = [torch.zeros(dim, device=device, dtype=dtype) for _ in range(args.num_workers)]
    fedac_optimizer_states = [None for _ in range(args.num_workers)]
    event_queue = []
    event_id = 0
    current_time = 0.0

    def dispatch(worker_id, now):
        nonlocal event_id
        snapshot = flat_params(server_model).detach().clone()
        use_cv = server.use_cv
        if args.alg == "fedac":
            if args.fedac_client_mode == "legacy":
                batch, worker_iters[worker_id] = next_batch(worker_loaders[worker_id], worker_iters[worker_id])
                payload = compute_worker_grad(
                    worker_model,
                    snapshot,
                    batch,
                    device,
                    dtype,
                    local_control=worker_controls[worker_id].clone() if use_cv else None,
                    global_control=server.global_control.clone() if use_cv else None,
                    grad_clip_norm=args.grad_clip_norm,
                    cv_momentum=args.cv_momentum,
                    cv_delta_clip_norm=args.cv_delta_clip_norm,
                    weight_decay=args.weight_decay,
                    label_smoothing=args.label_smoothing,
                )
                payload["snapshot"] = snapshot.detach().clone()
                finish_scale = 1.0
            else:
                if args.fedac_lr_mode == "shared":
                    fedac_local_lr = server.current_lr()
                else:
                    fedac_local_lr = server.current_fedac_local_lr()
                payload = compute_fedac_worker_payload(
                    worker_model,
                    snapshot,
                    worker_loaders[worker_id],
                    device,
                    dtype,
                    local_control=worker_controls[worker_id].clone() if use_cv else None,
                    global_control=server.global_control.clone() if use_cv else None,
                    local_lr=fedac_local_lr,
                    local_epochs=args.fedac_local_epochs,
                    local_momentum=args.fedac_local_momentum,
                    weight_decay=args.fedac_local_weight_decay,
                    optimizer_state=fedac_optimizer_states[worker_id] if args.fedac_persist_optimizer == 1 else None,
                    label_smoothing=args.label_smoothing,
                )
                finish_scale = float(max(1, args.fedac_local_epochs))
        else:
            batch, worker_iters[worker_id] = next_batch(worker_loaders[worker_id], worker_iters[worker_id])
            payload = compute_worker_grad(
                worker_model,
                snapshot,
                batch,
                device,
                dtype,
                local_control=worker_controls[worker_id].clone() if use_cv else None,
                global_control=server.global_control.clone() if use_cv else None,
                grad_clip_norm=args.grad_clip_norm,
                cv_momentum=args.cv_momentum,
                cv_delta_clip_norm=args.cv_delta_clip_norm,
                weight_decay=args.weight_decay,
                label_smoothing=args.label_smoothing,
            )
            payload["snapshot"] = snapshot.detach().clone()
            finish_scale = 1.0
        finish_time = now + worker_delay(worker_id, server.step, args) * finish_scale
        heapq.heappush(
            event_queue,
            (
                finish_time,
                event_id,
                worker_id,
                server.step,
                payload,
            ),
        )
        event_id += 1

    for worker_id in range(args.num_workers):
        dispatch(worker_id, current_time)

    results = {
        "args": dict(vars(args)),
        "reproducibility": build_reproducibility_metadata(args, device),
        "train_loss": [],
        "train_prec": [],
        "test_loss": [],
        "test_prec": [],
        "counts": [],
        "avg_staleness": [],
        "max_staleness": [],
        "server_control_norm": [],
        "status": {
            "completed_epochs": 0,
            "stopped_early": False,
            "early_abort_reason": None,
        },
    }
    server.counts.setdefault("batch_accept", 0)
    server.counts.setdefault("batch_reject", 0)

    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0
    epoch_staleness = []

    for update_idx in range(total_updates):
        finish_time, _, worker_id, dispatch_step, payload = heapq.heappop(event_queue)
        current_time = finish_time
        tau = server.step - dispatch_step
        if server.use_cv:
            if payload["new_local_control"] is not None:
                worker_controls[worker_id] = payload["new_local_control"].clone()
            if args.alg == "fedac":
                if args.fedac_client_mode == "exact" and args.fedac_persist_optimizer == 1:
                    fedac_optimizer_states[worker_id] = payload.get("optimizer_state")
                if args.fedac_client_mode == "legacy":
                    server.global_control.add_(payload["delta_control"], alpha=args.cv_server_lr / args.num_workers)
                    if args.cv_global_clip_norm > 0:
                        clipped = clip_vector_norm(server.global_control, args.cv_global_clip_norm)
                        server.global_control.copy_(clipped)
            if args.alg in ("asyncsgd_cv", "asyncsam_cv"):
                server.global_control.add_(payload["delta_control"], alpha=args.cv_server_lr / args.num_workers)
                if args.cv_global_clip_norm > 0:
                    clipped = clip_vector_norm(server.global_control, args.cv_global_clip_norm)
                    server.global_control.copy_(clipped)
        xk = flat_params(server_model).to(dtype)
        x_candidate, x_base = server.update(
            xk,
            payload["grad"],
            tau,
            snapshot=payload["snapshot"],
            delta_control=None if (args.alg == "fedac" and args.fedac_client_mode == "legacy") else payload["delta_control"],
            worker_delta=payload.get("worker_delta"),
            worker_id=worker_id,
        )
        x_next = x_candidate
        if args.alg in ("asyncsam", "asyncsam_cv", "asyncaa") and args.sam_batch_accept:
            cand_loss = batch_loss_for_snapshot(worker_model, x_candidate, payload["batch_cpu"], device)
            base_loss = batch_loss_for_snapshot(worker_model, x_base, payload["batch_cpu"], device)
            if cand_loss > base_loss * (1.0 + args.sam_batch_tol):
                x_next = x_base
                server.counts["batch_reject"] += 1
            else:
                server.counts["batch_accept"] += 1
        set_flat_params(server_model, x_next.to(next(server_model.parameters()).dtype))

        epoch_loss += payload["loss"] * payload["size"]
        epoch_correct += payload["correct"]
        epoch_total += payload["size"]
        epoch_staleness.append(tau)

        if update_idx % args.log_interval == 0:
            print(
                "Update {}/{} | alg={} | worker={} | tau={} | accepted={} fallback={}".format(
                    update_idx,
                    total_updates,
                    args.alg,
                    worker_id,
                    tau,
                    server.counts["accepted"],
                    server.counts["fallback"],
                )
            , flush=True)

        dispatch(worker_id, current_time)

        if (update_idx + 1) % updates_per_epoch == 0:
            epoch = (update_idx + 1) // updates_per_epoch
            train_loss = epoch_loss / epoch_total
            train_prec = epoch_correct / epoch_total
            print(
                "Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                    train_loss,
                    epoch_correct,
                    epoch_total,
                    100.0 * train_prec,
                )
            , flush=True)
            test_loss, test_prec = evaluate(server_model, device, test_loader)
            print(
                "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                    test_loss,
                    int(test_prec * len(test_subset)),
                    len(test_subset),
                    100.0 * test_prec,
                )
            , flush=True)

            results["train_loss"].append(float(train_loss))
            results["train_prec"].append(float(train_prec))
            results["test_loss"].append(float(test_loss))
            results["test_prec"].append(float(test_prec))
            results["counts"].append(dict(server.counts))
            results["avg_staleness"].append(float(np.mean(epoch_staleness)))
            results["max_staleness"].append(int(max(epoch_staleness)))
            results["server_control_norm"].append(float(torch.norm(server.global_control).item()))
            results["status"]["completed_epochs"] = int(epoch)
            update_summary(results)

            if args.partial_dump_every_epoch > 0 and (epoch % args.partial_dump_every_epoch == 0):
                write_result_artifacts(args.dump_data, results, partial=True)

            early_abort_reason = None
            if not math.isfinite(float(test_loss)):
                early_abort_reason = "non_finite_test_loss"
            elif args.early_abort_max_loss > 0 and float(test_loss) > args.early_abort_max_loss:
                early_abort_reason = "test_loss_above_threshold"
            elif (
                args.early_abort_epoch > 0
                and epoch >= args.early_abort_epoch
                and args.early_abort_min_acc >= 0
                and 100.0 * float(test_prec) < args.early_abort_min_acc
            ):
                early_abort_reason = "test_acc_below_threshold"

            if early_abort_reason is not None:
                results["status"]["stopped_early"] = True
                results["status"]["early_abort_reason"] = early_abort_reason
                print(
                    "Early abort triggered at epoch {} | reason={} | test_acc={:.2f}% | test_loss={:.4f}".format(
                        epoch,
                        early_abort_reason,
                        100.0 * float(test_prec),
                        float(test_loss),
                    ),
                    flush=True,
                )
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_total = 0
                epoch_staleness = []
                break

            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            epoch_staleness = []

    update_summary(results)
    write_result_artifacts(args.dump_data, results, partial=False)


if __name__ == "__main__":
    main()
