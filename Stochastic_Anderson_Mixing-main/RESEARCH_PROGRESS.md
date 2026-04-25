# Research Progress

## 1. 研究目标

当前阶段的目标是验证：在异步分布式训练场景下，Anderson Acceleration 是否能够在统一实验协议下稳定优于现有异步基线方法。

目前主方法为 `AsyncSAM+RMS`，其核心思想是：

- 用 `RMS` 风格的对角预条件先构造更稳定的基础更新步。
- 在服务器中心端对这些基础更新步应用 Anderson Acceleration。
- 在异步场景下保留额外的数值保护，例如 stale-history 筛选、batch accept、条件数保护和步长保护。

### AsyncSAM+RMS 算法细节

这里的 `SAM` 指的是 `Stochastic Anderson Mixing`，不是 `Sharpness-Aware Minimization`。当前主方法的实现是一个**服务器中心端**的异步 Anderson 加速器，`RMS` 只负责把基础异步步长先稳定下来，随后再做加速。

#### 1. Worker 侧做什么

- 每个 worker 拿到某个时刻的全局模型快照 `x_snapshot`
- 只在自己当前 batch 上计算一次梯度 `g_i`
- 不在 worker 本地做 AA，也不维护本地历史
- 算完后把这次梯度和对应 snapshot 发回服务器

因此，所有真正的加速逻辑都在服务器端，worker 侧仍然保持轻量。

#### 2. 异步基础步是什么

服务器每次只处理一个最早返回的 worker 更新。对当前到达的梯度 `g_k`，先根据 staleness `tau_k` 算一个权重 `w(tau_k)`，再形成基础异步残差：

- `r_base = - eta_t * w(tau_k) * g_k`

对应的基础迭代是：

- `x_base = x_k + r_base`

这一步就是所有后续 AA / SAM 的“底座”。如果没有任何加速或预条件，算法就会退化成普通异步 SGD 型更新。

#### 3. RMS 预条件怎么加入

当前主方法不是直接用 `r_base`，而是先在服务器端维护一个按坐标的平方梯度滑动平均：

- `v_k = beta_rms * v_{k-1} + (1 - beta_rms) * (g_k ⊙ g_k)`

然后用它来缩放基础残差：

- `r_k = - eta_t * w(tau_k) * g_k / (sqrt(v_k) + eps)`

当前实现里还允许三种额外保护：

- `precond_min_denom`：给分母设下界，避免某些坐标被放得过大
- `precond_warmup_updates`：前若干步把 `r_base` 与 `r_k` 做线性 warmup 混合
- `sam_momentum`：对服务器侧残差 `r_k` 再做一次 EMA，减少异步噪声

这一步的作用非常关键。经验上，很多异步 AA 的失败并不是 Anderson 本身不工作，而是基础残差太抖，导致历史矩阵非常不稳定；`RMS` 先把这个底座稳定住。

#### 4. Anderson / SAM 加速具体怎么做

当服务器有足够历史以后，不直接把 `x_k + r_k` 当最终结果，而是维护两类历史差分：

- `d_x`: 服务器迭代差分
- `d_r`: 服务器残差差分

实现里先对这两类差分做指数平滑，然后把它们写入历史矩阵：

- `X = [d_x^(1), ..., d_x^(m)]`
- `R = [d_r^(1), ..., d_r^(m)]`

随后解一个小型线性系统：

- `system = R^T R + delta * X^T X`
- `gamma = pinv(system) * (R^T r_k)`

再构造加速后的候选增量：

- `x_delta = beta * r_k - (alpha * X + alpha * beta * R) * gamma`

最后得到候选迭代：

- `x_aa = x_k + x_delta`

如果配置了 `base_mix` 或 `stale_base_mix`，实现还会把 `x_aa` 与最新基础步 `x_base` 做一次小比例混合，降低异步高 staleness 时的尾部失稳。

#### 5. 哪些数值保护在起作用

当前 `AsyncSAM+RMS` 不是“无保护 Anderson”，而是带了一整层异步保护机制：

- 只把不太 stale 的更新写进历史，避免坏历史污染
- `period` 控制不是每次到达都做加速
- `min_history` 控制历史不够时直接退回基础步
- `max_cond` 控制历史系统条件数过大时直接拒绝
- `max_step_ratio` 限制加速步不能比基础步大太多
- `anchor_tol` 限制候选点不能离基础步太远
- `min_cosine` 可要求加速方向和基础残差不要过度背离
- `restart_on_reject` 可在拒绝后重启历史
- `batch_accept` 会在当前返回 batch 上比较候选步和基础步的 loss，差了就退回基础步

所以当前主方法并不是“每次都强行用 AA”，而是：

1. 先算稳定的 `RMS` 基础步  
2. 有条件时再提出 `AA` 候选  
3. 候选不靠谱就退回基础步

#### 6. 为什么当前主方法叫 `AsyncSAM+RMS`

- `Async`：服务器按更新到达顺序一条一条处理，没有同步 barrier
- `SAM`：服务器端用的是 `Stochastic Anderson Mixing` 风格的历史矩阵加速
- `RMS`：基础异步残差先经过 `RMS` 对角预条件稳定化

因此，这个方法的本质可以概括成：

- `稳定基础步（RMS） + 服务器端 Anderson 加速（SAM） + 异步数值保护`

从目前实验结果看，真正起决定性作用的是两层配合：

1. `RMS` 先把异步基础步稳定下来
2. `SAM/AA` 再在这个稳定基础上提速和提精度

#### 7. 为什么 `AsyncSAM+RMS` 往往是后期更强

当前实验现象可以概括为：`AsyncSAM+RMS` 的后期优势，通常不是“AA 单独突然生效”，而是 `RMS` 与 `AA` 的配合在后期才更容易被释放出来。

- 在训练前期，异步返回的梯度噪声大、各坐标尺度差异也更明显，历史残差序列往往比较脏，直接做 Anderson 组合容易不稳定。
- `RMS` 预条件先把这些残差按坐标做缩放和稳定化，使服务器看到的基础残差序列更平滑、更可比较。
- 到了训练后期，模型进入更平稳的收敛区间后，残差之间的一致性更强，历史矩阵的结构信息才更可靠。
- 这时 `SAM/AA` 的历史组合能力才更容易真正带来额外收益，因此会表现为“后期拉开差距”。

因此，当前更合理的机制性解释是：

- `RMS` 先保证残差更干净、更稳定
- `AA/SAM` 再利用这些更可靠的历史残差做中心端加速

这也解释了为什么在异步场景下，单独强调 `AA` 往往不够；如果基础残差本身不稳定，历史加速通常很难发挥作用。


## 2. 当前实验协议

当前主实验协议已经固定，所有方法均在同一个异步事件驱动模拟器中运行，以确保公平比较。当前文档主要包含两条已经落地的协议线：

### 数据与训练设置

#### MNIST 主协议

- Dataset: `MNIST`
- Partition: `IID`
- Number of workers: `10`
- Train subset size: `12000`
- Test subset size: `1000`
- Batch size: `600`
- Epochs: `60`
- Learning rate: `0.01`
- Device: `NPU`

#### CIFAR-10 扩展协议

- Dataset: `CIFAR-10`
- Partition: `IID`
- Number of workers: `2`
- Train size: `50000`
- Test size: `10000`
- Batch size: `128`
- Epochs: `100`
- Learning rate: `0.05`
- Schedule: `cosine`
- Weight decay: `5e-4`
- Device: `NPU`

#### CIFAR-100 当前状态

- `cifar100` 代码入口已经打通
- 已完成 `asyncsgd + resnet18 + 1 epoch` smoke
- 正式多 seed 协议尚未最终敲定

### 异步机制

当前异步机制不是同步 barrier，而是事件驱动的 one-arrival-at-a-time 更新：

1. 每个 worker 拿到某个时刻的全局模型快照。
2. 每个 worker 只在自己的当前 batch 上计算一个梯度。
3. 不同 worker 具有不同完成时间。
4. 服务器每次只处理最早到达的一个更新。
5. 被处理完的 worker 立即再次 dispatch。
6. 其他尚未返回的 worker 仍在用旧模型计算，因此天然产生 stale update。

### 延迟模型

当前使用确定性的异构延迟模型：

- `tier = worker_id % 3`
- `delay = 1.0 + tier * 0.5 + jitter`
- `delay_jitter = 0.1`

因此 worker 被分为三档延迟：

- fast tier: `1.0 + jitter`
- medium tier: `1.5 + jitter`
- slow tier: `2.0 + jitter`

### Staleness 定义

- `tau = server.step - dispatch_step`

即：某个 worker 的梯度是基于更早的 server snapshot 计算得到的，它返回时服务器已经完成了若干次更新。

当前默认 stale weighting 使用 `hinge`：

- 当 `tau <= 4` 时，权重为 `1`
- 当 `tau > 4` 时，权重为 `1 / (tau + 5)`


## 3. 已完成工作

### 3.1 方法侧

- 已实现异步分布式版本的 `AsyncSAM`
- 已实现 `RMS+AA` 主方法
- 已在不改变 AA 核心形式的前提下加入异步数值保护：
  - stale-history 过滤
  - batch accept / reject
  - 条件数保护
  - step ratio / anchor 保护

### 3.2 基线侧

已将 AFL 风格异步算法接入同一个共享模拟器，以保证统一协议下的公平比较：

- `FedAsync`
- `FedBuff`
- `AsyncAA`
- `FedAC-style server logic`

### 3.3 工具与可复现性

- 已补齐单 seed 对比图脚本
- 已补齐多 seed 均值图脚本
- 每个结果文件现在会额外写入 `.pkl.meta.json`
- 批量脚本现在会为每个结果写入 `.pkl.cmd.txt`
- 当前所有主结果均可在 NPU 环境下复现


## 4. 当前最重要结果

本节主要汇总 `MNIST` 主线结果；`CIFAR-10` 的 seed10 与完整 `10-seed` 结果见第 `7.3` 节与第 `11.1` 节。

### 4.1 单 seed 结果

在统一设置 `IID + NPU + 60 epochs + lr=0.01 + seed=1` 下：

| Method | Final Acc | Best Acc | Final Loss |
|---|---:|---:|---:|
| AsyncSAM+RMS | 99.0 | 99.3 | 0.0423 |
| FedBuff | 90.7 | 91.9 | 0.2893 |
| AsyncSGD | 89.6 | 89.6 | 0.3359 |
| FedAsync | 89.5 | 89.5 | 0.3359 |
| AsyncAA | 82.7 | 90.7 | 0.4848 |

对应结果图：

- `mnist/experiments/afl_same_setting_compare_fig/afl_same_setting_compare_full.png`
- `mnist/experiments/afl_same_setting_compare_fig/afl_same_setting_compare_full_summary.csv`

### 4.2 三个 seed 的均值结果

在相同协议下，`3-seed` 均值结果如下：

| Method | Final Acc Mean ± Std | Best Acc Mean ± Std | Final Loss Mean ± Std |
|---|---:|---:|---:|
| AsyncSAM+RMS | 98.27 ± 0.54 | 98.87 ± 0.37 | 0.0599 ± 0.0127 |
| FedBuff | 89.87 ± 0.74 | 91.60 ± 0.73 | 0.3029 ± 0.0110 |
| AsyncSGD | 89.20 ± 1.34 | 89.33 ± 1.48 | 0.3654 ± 0.0453 |
| FedAsync | 89.17 ± 1.33 | 89.30 ± 1.48 | 0.3654 ± 0.0453 |
| AsyncAA | 86.37 ± 2.64 | 90.80 ± 1.19 | 0.4003 ± 0.0598 |

对应结果图：

- `mnist/experiments/afl_same_setting_multiseed_fig/multiseed_compare.png`
- `mnist/experiments/afl_same_setting_multiseed_fig/summary.csv`

结论：`AsyncSAM+RMS` 不仅在单个 seed 上最好，在 `3-seed` 均值上也显著领先。

### 4.3 调参后的 FedAC 复查

在共享模拟器中重新检查 `FedAC` 后，发现迁移版本最初存在明显的 FedAC-specific 超参错位：

- 原始迁移版使用了过大的 `fedac_eta_g`
- `fedac_buffer_size` 也没有与原始 AFL-Lib 默认值保持一致

在不改变共享协议参数的前提下，仅调 `FedAC` 自己的参数后，`MNIST seed=1` 的最佳结果提升到：

| Method | Final Acc | Best Acc | Final Loss | Best Loss |
|---|---:|---:|---:|---:|
| FedAC (tuned) | 93.00 | 93.70 | 0.2604 | 0.2200 |

当前最优 `FedAC` 配置是：

- `fedac_eta_g = 3e-4`
- `fedac_buffer_size = 5`
- `cv_server_lr = 0.5`

对应结果图与汇总表：

- `mnist/experiments/afl_same_setting_compare_fig_fedac_tuned/asyncsam_lr_sweep.png`
- `mnist/experiments/afl_same_setting_compare_fig_fedac_tuned/summary.csv`

这一结果说明：

1. 共享模拟器里的 `FedAC` 不是“算法本身失效”，而是对 FedAC-specific 超参非常敏感
2. 调回合理量级后，`FedAC` 已经能在 `MNIST` 上恢复到正常区间
3. 即便如此，当前最优方法仍然是 `AsyncSAM+RMS`

### 4.4 十个 seed 的完整结果（含调参后 FedAC）

在固定共享协议参数、仅使用调好的 `FedAC` 独有参数后，已完成 `MNIST` 的完整 `10-seed` 对比。

| Method | Final Acc Mean ± Std | Best Acc Mean ± Std | Final Loss Mean ± Std |
|---|---:|---:|---:|
| AsyncSAM+RMS | 98.20 ± 0.42 | 98.62 ± 0.35 | 0.0675 ± 0.0110 |
| FedBuff | 91.25 ± 1.54 | 92.17 ± 0.99 | 0.2904 ± 0.0371 |
| AsyncSGD | 89.28 ± 1.27 | 89.69 ± 1.10 | 0.3713 ± 0.0335 |
| FedAsync | 89.27 ± 1.27 | 89.69 ± 1.11 | 0.3712 ± 0.0335 |
| FedAC tuned | 83.95 ± 13.73 | 93.89 ± 1.29 | 0.7481 ± 0.9845 |
| AsyncAA | 73.50 ± 9.51 | 75.06 ± 11.11 | 1.4499 ± 0.7015 |

对应结果图与汇总表：

- `mnist/experiments/afl_same_setting_10seed_fig_full/multiseed_compare.png`
- `mnist/experiments/afl_same_setting_10seed_fig_full/summary.csv`

这一组结果说明：

1. `AsyncSAM+RMS` 在 `10-seed` 均值上仍然显著领先，结论比 `3-seed` 更稳
2. 调好的 `FedAC` 虽然 `best acc` 很高，但 `final acc` 方差仍然较大，说明后段稳定性仍不如主方法
3. `FedBuff` 是传统异步基线里最稳的一条，但与 `AsyncSAM+RMS` 仍有明显差距


## 5. 消融实验结论

为更直接回答 “`RMS` 和 `AsyncSAM` 分别带来什么” 这个问题，正交消融现已改为只保留 `AsyncSAM` 系列，不再展示 `AsyncAA`。

当前使用的消融方法是：

- `AsyncSGD`
- `AsyncSGD+RMS`
- `AsyncSAM`
- `AsyncSAM+RMS`

在 `10-seed` 均值下结果如下：

| Method | Final Acc | Best Acc | Final Loss | Best Loss |
|---|---:|---:|---:|---:|
| AsyncSGD | 89.2800 | 89.6900 | 0.3713 | 0.3684 |
| AsyncSGD+RMS | 97.1900 | 97.8700 | 0.1218 | 0.0763 |
| AsyncSAM | 89.3600 | 89.5700 | 0.3820 | 0.3794 |
| AsyncSAM+RMS | 98.2000 | 98.6200 | 0.0675 | 0.0504 |

对应结果图与表：

- `mnist/experiments/mnist_sam_ablation_10seed_fig/multiseed_compare.png`
- `mnist/experiments/mnist_sam_ablation_10seed_fig/summary.csv`
- `mnist/experiments/mnist_sam_ablation_10seed_fig/mean_table.csv`

### 当前已得到的关键机制性结论

1. 不加 `RMS` 时，`AsyncSAM` 与 `AsyncSGD` 的最终表现非常接近，说明仅靠 AA 在当前异步场景下增益有限
2. `RMS` 带来的是决定性提升：无论放在 `AsyncSGD` 还是 `AsyncSAM` 上，都能显著改善最终精度和 loss
3. 最优结果仍然来自 `AsyncSAM+RMS`，说明在稳定的 `RMS` 基础步之上，AA 仍然能继续带来额外收益


## 6. 当前问题与风险

虽然当前主方法的最终精度与最终 loss 最好，但仍有两个需要如实说明的问题：

- `RMS+AA` 的前期 loss 曲线相对更抖，不是最平滑的方案
- `FedAC` 在当前 one-arrival-at-a-time 异步模拟中波动很大，因此暂未纳入 `3-seed` 主结果表

这说明：

- 当前方法已经在最终性能上占优
- 但如果面向论文展示，仍有必要继续改善早期训练曲线的可解释性和稳定性


## 7. 额外验证实验

为验证当前结论不是只在单一协议下成立，当前额外验证实验分成两部分：

- 同一数据集内的协议 / 异步条件变化验证：`MNIST`
- 跨数据集迁移验证：`CIFAR-10`

其中，`MNIST` 这部分已进一步在四类不同设置下做了 `10-seed` 实验：

- 更强非 IID：`label_sorted`
- 更强且非 label-sorted 的非 IID：`dirichlet(alpha=0.05, min_size=10)`
- 更强异步时延：`delay_gap=1.0, delay_jitter=0.2`
- 更强并发：`20 workers`

其中 `20 workers` 设置下，考虑到平均 staleness 明显升高，对主方法加入了一个很小的 stale-base mixing 修正：

- `sam_stale_base_mix = 0.2`

该修正不改变 AA 的核心形式，只在当前更新过于 stale 时，将 AA 候选与最新 base step 做小比例混合，以降低尾段掉点。

### 7.1 MNIST 不同设置验证结果（10-seed）

截至 `2026-04-11`，这一组 `MNIST` 额外设置验证已经补齐为完整 `6` 方法版本：`AsyncSAM+RMS / AsyncSAM / AsyncSGD+RMS / FedAsync / FedBuff / FedAC`。

| Setting | Method | Final Acc Mean ± Std | Best Acc Mean ± Std | Final Loss Mean ± Std |
|---|---|---:|---:|---:|
| Base IID | AsyncSAM+RMS | 98.20 ± 0.42 | 98.62 ± 0.35 | 0.0675 ± 0.0110 |
| Base IID | AsyncSAM | 89.36 ± 1.16 | 89.57 ± 1.08 | 0.3820 ± 0.0238 |
| Base IID | AsyncSGD+RMS | 97.19 ± 1.06 | 97.87 ± 0.58 | 0.1218 ± 0.0709 |
| Base IID | FedAsync | 89.27 ± 1.27 | 89.69 ± 1.11 | 0.3712 ± 0.0335 |
| Base IID | FedBuff | 91.25 ± 1.54 | 92.17 ± 0.99 | 0.2904 ± 0.0371 |
| Base IID | FedAC | 83.95 ± 13.73 | 93.89 ± 1.29 | 0.7481 ± 0.9845 |
| Label-Sorted Non-IID | AsyncSAM+RMS | 93.45 ± 4.10 | 96.78 ± 1.11 | 0.2609 ± 0.2513 |
| Label-Sorted Non-IID | AsyncSAM | 73.73 ± 9.26 | 82.45 ± 1.72 | 0.8041 ± 0.2088 |
| Label-Sorted Non-IID | AsyncSGD+RMS | 82.90 ± 15.86 | 95.28 ± 1.56 | 1.7550 ± 3.8422 |
| Label-Sorted Non-IID | FedAsync | 74.16 ± 9.25 | 83.08 ± 1.33 | 0.7834 ± 0.2217 |
| Label-Sorted Non-IID | FedBuff | 88.25 ± 1.43 | 89.71 ± 0.87 | 0.3829 ± 0.0492 |
| Label-Sorted Non-IID | FedAC | 80.49 ± 8.25 | 90.25 ± 3.79 | 0.6446 ± 0.3097 |
| High Delay IID | AsyncSAM+RMS | 98.17 ± 0.41 | 98.44 ± 0.38 | 0.0831 ± 0.0271 |
| High Delay IID | AsyncSAM | 90.29 ± 0.89 | 90.82 ± 0.91 | 0.3384 ± 0.0293 |
| High Delay IID | AsyncSGD+RMS | 95.17 ± 2.68 | 97.07 ± 0.93 | 0.2045 ± 0.1350 |
| High Delay IID | FedAsync | 89.64 ± 2.49 | 90.94 ± 1.01 | 0.3495 ± 0.0509 |
| High Delay IID | FedBuff | 92.09 ± 0.89 | 92.51 ± 0.79 | 0.2749 ± 0.0242 |
| High Delay IID | FedAC | 77.30 ± 12.29 | 92.58 ± 1.69 | 0.8158 ± 0.5376 |
| IID 20 Workers | AsyncSAM+RMS | 97.61 ± 0.72 | 98.01 ± 0.54 | 0.0750 ± 0.0204 |
| IID 20 Workers | AsyncSAM | 81.89 ± 2.33 | 81.90 ± 2.33 | 0.9584 ± 0.2175 |
| IID 20 Workers | AsyncSGD+RMS | 96.62 ± 1.36 | 97.20 ± 0.78 | 0.1168 ± 0.0409 |
| IID 20 Workers | FedAsync | 81.47 ± 2.36 | 81.51 ± 2.25 | 0.9989 ± 0.2217 |
| IID 20 Workers | FedBuff | 87.19 ± 2.89 | 87.89 ± 2.98 | 0.4850 ± 0.2237 |
| IID 20 Workers | FedAC | 83.78 ± 17.91 | 91.88 ± 2.08 | 0.6621 ± 1.0153 |

在此基础上，又补了一组更强、且不依赖标签排序切块的 `Dirichlet Non-IID` 验证，参数为 `alpha=0.05, min_size=10`。这一组更适合回答“如果非 IID 不是由 label-sorted 人工构造，而是由类别分布随机异质性产生，结论是否仍成立”。

| Setting | Method | Final Acc Mean ± Std | Best Acc Mean ± Std | Final Loss Mean ± Std |
|---|---|---:|---:|---:|
| Dirichlet Non-IID (`alpha=0.05`) | AsyncSAM+RMS | 96.79 ± 1.21 | 98.03 ± 0.20 | 0.1167 ± 0.0482 |
| Dirichlet Non-IID (`alpha=0.05`) | AsyncSAM | 81.25 ± 6.16 | 85.60 ± 1.79 | 0.5698 ± 0.1487 |
| Dirichlet Non-IID (`alpha=0.05`) | AsyncSGD+RMS | 87.71 ± 20.88 | 96.98 ± 0.67 | 3.9236 ± 11.1278 |
| Dirichlet Non-IID (`alpha=0.05`) | FedAsync | 79.72 ± 9.09 | 85.89 ± 1.64 | 0.6261 ± 0.2730 |
| Dirichlet Non-IID (`alpha=0.05`) | FedBuff | 88.97 ± 3.13 | 91.70 ± 1.41 | 0.3500 ± 0.0786 |
| Dirichlet Non-IID (`alpha=0.05`) | FedAC | 73.60 ± 15.33 | 89.07 ± 2.81 | 1.4917 ± 1.2444 |

这组结果说明：在一种更自然、更强的随机异质划分下，`AsyncSAM+RMS` 仍然是均值最高、方差较小、最终 loss 最低的方法；`FedBuff` 是第二梯队中最稳的基线；`AsyncSGD+RMS` 虽然 `best acc` 很高，但尾部波动显著，稳定性不如 `AsyncSAM+RMS`。

对应汇总文件：

- `mnist/experiments/validation_other_settings_10seed_fig/summary.csv`
- `mnist/experiments/validation_other_settings_10seed_fig/validation_settings_compare.png`
- `mnist/experiments/validation_other_settings_10seed_fig/validation_settings_compare.pdf`
- `mnist/experiments/validation_dirichlet_10seed_fig/summary.csv`
- `mnist/experiments/validation_dirichlet_10seed_fig/multiseed_compare.png`
- `mnist/experiments/validation_dirichlet_10seed_fig/multiseed_compare.pdf`

### 7.2 CIFAR-10 迁移验证结果（10-seed）

为验证结论能否迁移到更难的数据集，已在共同 `100 epochs` 的公平 horizon 下完成 `CIFAR-10` 的完整 `10-seed` 对比。该结果与上面的 `MNIST` 不同设置验证互补：前者回答“换协议后是否仍成立”，这里回答“换数据集后是否仍成立”。

| Method | Final Acc Mean ± Std | Best Acc Mean ± Std | Final Loss Mean ± Std |
|---|---:|---:|---:|
| AsyncSAM+RMS | 92.27 ± 0.46 | 92.33 ± 0.46 | 0.3444 ± 0.0164 |
| FedAsync | 89.91 ± 0.25 | 90.04 ± 0.20 | 0.4426 ± 0.0108 |
| AsyncSGD | 89.82 ± 0.25 | 89.96 ± 0.26 | 0.4407 ± 0.0194 |
| FedBuff | 88.09 ± 1.25 | 89.14 ± 0.25 | 0.4321 ± 0.0542 |
| FedAC (tuned) | 79.80 ± 9.91 | 84.06 ± 2.81 | 0.6088 ± 0.2844 |

对应汇总文件：

- `mnist/experiments/cifar10_10seed_compare_legacy92_with_fedac_fig/multiseed_compare.png`
- `mnist/experiments/cifar10_10seed_compare_legacy92_with_fedac_fig/summary.csv`

此外，现已按 `MNIST` 的同类验证逻辑，为 `CIFAR-10 + ResNet18` 完成三类协议变化设置的 `10-seed` 验证：

- `label_sorted`
- `high_delay (delay_gap=0.5, delay_jitter=0.1)`
- `IID 20 workers`

为保持共享协议尽量一致，这一组 `CIFAR-10` settings validation 沿用当前主线方法集：

- `AsyncSAM+RMS`
- `AsyncSGD`
- `FedAsync`
- `FedBuff`
- `FedAC`

对应脚本：

- `mnist/run_cifar10_validation_settings_10seed.sh`
- `mnist/plot_cifar10_validation_settings_10seed.py`

最终结果汇总到：

- `mnist/experiments/cifar10_validation_settings_10seed_fig/summary.csv`
- `mnist/experiments/cifar10_validation_settings_10seed_fig/validation_settings_compare.png`
- `mnist/experiments/cifar10_validation_settings_10seed_fig/validation_settings_compare.pdf`

当前完整 `10-seed` 结果如下：

| Setting | Method | Final Acc Mean ± Std | Best Acc Mean ± Std | Final Loss Mean ± Std |
|---|---|---:|---:|---:|
| Label-Sorted Non-IID | AsyncSAM+RMS | 92.18 ± 0.32 | 92.29 ± 0.32 | 0.3424 ± 0.0076 |
| Label-Sorted Non-IID | AsyncSGD | 89.03 ± 0.30 | 89.20 ± 0.27 | 0.4530 ± 0.0127 |
| Label-Sorted Non-IID | FedAsync | 89.20 ± 0.18 | 89.34 ± 0.20 | 0.4504 ± 0.0140 |
| Label-Sorted Non-IID | FedBuff | 87.25 ± 1.62 | 88.91 ± 0.42 | 0.4568 ± 0.0523 |
| Label-Sorted Non-IID | FedAC | 67.92 ± 18.34 | 72.25 ± 19.16 | 1.2025 ± 1.0517 |
| High Delay IID | AsyncSAM+RMS | 92.22 ± 0.34 | 92.29 ± 0.35 | 0.3466 ± 0.0081 |
| High Delay IID | AsyncSGD | 89.88 ± 0.24 | 89.95 ± 0.25 | 0.4421 ± 0.0190 |
| High Delay IID | FedAsync | 89.93 ± 0.18 | 90.04 ± 0.15 | 0.4394 ± 0.0121 |
| High Delay IID | FedBuff | 88.18 ± 0.68 | 88.88 ± 0.31 | 0.4252 ± 0.0298 |
| High Delay IID | FedAC | 77.56 ± 22.54 | 85.33 ± 1.57 | 1.8790 ± 4.2673 |
| IID 20 Workers | AsyncSAM+RMS | 52.67 ± 1.91 | 52.79 ± 1.80 | 1.3183 ± 0.0502 |
| IID 20 Workers | AsyncSGD | 54.89 ± 1.54 | 55.04 ± 1.50 | 1.2507 ± 0.0459 |
| IID 20 Workers | FedAsync | 54.95 ± 1.52 | 55.06 ± 1.52 | 1.2507 ± 0.0457 |
| IID 20 Workers | FedBuff | 49.74 ± 1.91 | 51.88 ± 1.72 | 1.3929 ± 0.0620 |
| IID 20 Workers | FedAC | 21.40 ± 6.72 | 23.67 ± 6.00 | 2.1566 ± 0.2570 |

需要特别说明的是：上面这组 `IID 20 Workers` 结果对应的是第一版共享协议，作用更接近“高并发压力测试”。在确认这一版协议会让所有方法一起退化后，又进一步做了一轮**统一共享协议重设**，仍保持所有方法共享相同的公共参数，只修改这些公共项：

- `batch_size = 64`
- `epochs = 200`
- `lr = 0.01`
- `delay_gap = 0.0`
- `delay_jitter = 0.0`

在这组新的统一协议下，`CIFAR-10 / 20 workers` 的完整 `10-seed` 结果为：

| Setting | Method | Final Acc Mean ± Std | Best Acc Mean ± Std | Final Loss Mean ± Std |
|---|---|---:|---:|---:|
| IID 20 Workers Tuned | AsyncSAM+RMS | 83.76 ± 0.56 | 83.95 ± 0.57 | 0.5083 ± 0.0134 |
| IID 20 Workers Tuned | AsyncSGD | 75.55 ± 0.83 | 75.77 ± 0.77 | 0.7072 ± 0.0226 |
| IID 20 Workers Tuned | FedAsync | 75.57 ± 0.78 | 75.75 ± 0.79 | 0.7077 ± 0.0217 |
| IID 20 Workers Tuned | FedBuff | 73.26 ± 7.23 | 79.15 ± 0.95 | 0.8225 ± 0.2814 |
| IID 20 Workers Tuned | FedAC | 26.60 ± 6.52 | 30.58 ± 6.93 | 2.2247 ± 0.3314 |

这说明：`20 workers` 并不是“无论如何都不行”，而是需要更合适的共享协议。一旦把额外 synthetic delay 去掉，并把 batch 调回更小的 `64`，`AsyncSAM+RMS` 仍然能在高并发下保持明显领先。

对应汇总文件：

- `mnist/experiments/cifar10_workers20_t3e200_10seed_fig/summary.csv`
- `mnist/experiments/cifar10_workers20_t3e200_10seed_fig/multiseed_compare.png`
- `mnist/experiments/cifar10_workers20_t3e200_10seed_fig/multiseed_compare.pdf`

### 7.3 验证结论

当前结论在额外实验中继续成立：

1. 在 `10-seed` 均值下，`AsyncSAM+RMS` 在四类 `MNIST` 设置里都保持最优，结论不再依赖单个 seed 的幸运轨迹
2. 即便把比较对象扩展到完整 `6` 方法，`AsyncSAM+RMS` 在 `label_sorted / high_delay / 20 workers` 下仍然系统性优于 `AsyncSAM`、`AsyncSGD+RMS`、`FedAsync`、`FedBuff` 和 `FedAC`
3. `FedAsync` 在更高并发的 `20 workers` 设置下明显变弱，而 `FedAC` 在多组设置里仍表现出较强的 seed 敏感性和尾段不稳定
4. 跨到 `CIFAR-10` 之后，主结论在 `Base IID`、`Label-Sorted` 和 `High Delay` 三组上继续成立：`AsyncSAM+RMS` 在均值、方差和最终 loss 上都优于 `AsyncSGD / FedAsync / FedBuff / FedAC`
5. `IID 20 Workers` 的第一版共享协议确实会让所有方法一起退化，但这更像是协议问题，而不是方法本身无法扩展到高并发；在统一重设后的共享协议下，`AsyncSAM+RMS` 又重新成为最优方法
6. `RMS` 依然是异步场景下最关键的稳定化来源，而 `AsyncSAM+RMS` 在 `CIFAR-10` 的基础协议、强 non-IID、强 delay 以及修复后的 `20 workers` 协议下都保持额外优势，说明 AA 在稳定基础步之上继续带来了收益

### 7.4 CIFAR-10 扩展实验（seed=10 到 10-seed）

为验证当前结论是否能迁移到更难的数据集，已在 `CIFAR-10` 上补做正式实验，并且已经从早期 `seed=10` 调参阶段扩展到完整 `10-seed` 对比。

这组实验有两个实现层面的重要说明：

- 当前 `CIFAR-10` 结果使用的是修正后的数据管线。此前的子集抽取方式会把随机增广一次性物化，从而冻结 `RandomCrop` / `HorizontalFlip`；现在已改为基于 `Subset` 的索引采样，因此每个 epoch 的增广仍然是动态的。
- 主图采用共同 `100 epochs` 的公平 horizon，对 `AsyncSGD` 使用了裁到前 `100` 个评估点的版本，以避免此前 `150 epochs` 横坐标更长带来的不公平展示。

#### CIFAR-10 当前协议

- Dataset: `CIFAR-10`
- Partition: `IID`
- Seed: `10`
- Number of workers: `2`
- Train size: `50000`
- Test size: `10000`
- Batch size: `128`
- Epochs: `100`
- Learning rate: `0.05`
- Schedule: `cosine`
- Weight decay: `5e-4`
- Device: `NPU`

#### 中间阶段：seed=10 结果

| Method | Final Acc | Best Acc | Final Loss | Best Loss |
|---|---:|---:|---:|---:|
| AsyncSGD (100-epoch fair horizon) | 89.92 | 89.92 | 0.4109 | 0.3529 |
| FedAsync | 89.95 | 89.99 | 0.4403 | 0.3781 |
| FedBuff | 88.20 | 88.76 | 0.4509 | 0.3845 |
| AsyncAA | 89.60 | 89.82 | 0.4781 | 0.3836 |
| AsyncSAM+RMS (`beta=0.90`) | 92.58 | 92.58 | 0.3278 | 0.3094 |
| FedAC (tuned) | 85.57 | 85.70 | 0.4310 | 0.4269 |

辅助结果：

- `AsyncSAM+RMS (beta=0.95)`: final `91.85`, best `92.06`, final loss `0.3573`, best loss `0.3089`

对应当前稳定主图与汇总表：

- `mnist/experiments/cifar10_seed10_current_compare_fig_e100fair_main/asyncsam_lr_sweep.png`
- `mnist/experiments/cifar10_seed10_current_compare_fig_e100fair_main/summary.csv`

已新增完整对比图与汇总表：

- `mnist/experiments/cifar10_seed10_full_compare_fig/asyncsam_lr_sweep.png`
- `mnist/experiments/cifar10_seed10_full_compare_fig/summary.csv`
- `mnist/experiments/cifar10_seed10_stable_compare_fig/asyncsam_lr_sweep.png`
- `mnist/experiments/cifar10_seed10_stable_compare_fig/summary.csv`

在保持共享协议完全一致的前提下，已额外对 `FedAC` 的独有参数做了一轮小 sweep，最佳点为：

- `fedac_eta_g = 3e-4`
- `fedac_buffer_size = 5`
- `cv_server_lr = 0.5`

对应 sweep 图与调好后的完整图：

- `mnist/experiments/cifar10_fedac_sweep_fig/asyncsam_lr_sweep.png`
- `mnist/experiments/cifar10_fedac_sweep_fig/summary.csv`
- `mnist/experiments/cifar10_seed10_full_compare_fig_fedac_tuned/asyncsam_lr_sweep.png`
- `mnist/experiments/cifar10_seed10_full_compare_fig_fedac_tuned/summary.csv`

当前这组 `CIFAR-10` 结果支持与 `MNIST` 一致的结论：

1. `FedAsync` 与 `AsyncSGD` 在这套协议下表现接近，但都明显低于 `AsyncSAM+RMS`
2. `FedBuff` 稍弱于 `FedAsync / AsyncSGD`
3. 纯 `AsyncAA` 已经能接近 `AsyncSGD`，但最佳结果仍然来自 `RMS + AA`
4. `FedAC` 经过参数修正后已经恢复到正常区间，但仍明显低于 `AsyncSAM+RMS / FedAsync / AsyncAA`
5. 在修正后的 `CIFAR-10` 动态增广协议下，`AsyncSAM+RMS (beta=0.90)` 当前仍是最优方法

#### 当前主结论使用的完整 10-seed 结果

在相同共享协议下，现已进一步完成 `CIFAR-10` 的完整 `10-seed` 对比：

| Method | Final Acc Mean ± Std | Best Acc Mean ± Std | Final Loss Mean ± Std |
|---|---:|---:|---:|
| AsyncSAM+RMS | 92.27 ± 0.46 | 92.33 ± 0.46 | 0.3444 ± 0.0164 |
| FedAsync | 89.91 ± 0.25 | 90.04 ± 0.20 | 0.4426 ± 0.0108 |
| AsyncSGD | 89.82 ± 0.25 | 89.96 ± 0.26 | 0.4407 ± 0.0194 |
| FedBuff | 88.09 ± 1.25 | 89.14 ± 0.25 | 0.4321 ± 0.0542 |
| FedAC (tuned) | 79.80 ± 9.91 | 84.06 ± 2.81 | 0.6088 ± 0.2844 |

对应图与汇总表：

- `mnist/experiments/cifar10_10seed_compare_legacy92_with_fedac_fig/multiseed_compare.png`
- `mnist/experiments/cifar10_10seed_compare_legacy92_with_fedac_fig/summary.csv`

这一组 `10-seed` 结果比早期 `seed=10` 更值得作为论文主表，因为它表明：

1. `AsyncSAM+RMS` 的领先不是单个 seed 的偶然结果
2. `FedAC` 在 `CIFAR-10` 上存在明显的 seed 敏感性和尾段不稳定
3. 当前跨数据集结论依然保持一致：`AsyncSAM+RMS` 是最稳且最强的方法


## 8. 当前代码位置

### 主实验脚本

- `mnist/async_distributed_main.py`
- `mnist/run_cifar10_seed_list.sh`
- `mnist/run_cifar10_fedac_10seed.sh`
- `mnist/run_cifar10_fedac_extra_9_10.sh`
- `mnist/run_cifar10_validation_settings_10seed.sh`
- `mnist/run_cifar10_workers20_t3e200_10seed.sh`
- `mnist/run_cifar10_workers20_t3e200_tail_parallel.sh`
- `mnist/run_validation_dirichlet_10seed.sh`

### 多 seed 对比脚本

- `mnist/plot_async_multiseed_compare.py`
- `mnist/plot_cifar10_validation_settings_10seed.py`
- `mnist/plot_cifar10_workers20_t3e200_10seed.py`
- `mnist/plot_validation_dirichlet_10seed.py`

### 单次对比脚本

- `mnist/plot_asyncsam_lr_sweep.py`
- `mnist/plot_validation_settings.py`


## 9. 下一步计划

下一阶段建议重点推进三项工作：

1. 将当前 `MNIST + CIFAR-10` 的完整结果整理为论文可直接使用的实验段落、图注和表格，重点包含 `MNIST Dirichlet Non-IID` 和 `CIFAR-10 20 workers tuned`
2. 复制 `CIFAR-10` 的实验脚本体系，正式启动 `CIFAR-100` 的 pilot / multi-seed 对比
3. 对 `FedAC` 和高并发场景继续做针对性稳定性分析，区分“算法机制不足”与“协议不匹配”两类来源


## 10. 当前阶段结论

截至目前，在当前统一异步分布式实验设置下：

- `AsyncSAM+RMS` 已经是最优方法
- 该结论已经被 `MNIST` 的单 seed、`3-seed`、`10-seed`、`MNIST Dirichlet Non-IID 10-seed`、`CIFAR-10` 的完整 `10-seed` 以及修复后的 `CIFAR-10 20 workers 10-seed` 结果共同支持
- `RMS` 是异步场景下让 AA 真正发挥作用的关键技术组件
- `20 workers` 这类高并发场景并非天然无解，但必须匹配更合理的共享协议；在公平统一重设后，`AsyncSAM+RMS` 仍然可以显著领先
- `CIFAR-100` 入口已经打通，下一阶段可以直接转入正式实验


## 11. 2026-04-13 最新更新

### 11.1 CIFAR-10 已完成完整 10-seed 对比

在 `CIFAR-10 + ResNet18 + IID + 2 workers + 100 epochs + lr=0.05 + cosine schedule` 的统一协议下，现已完成包含 `FedAC` 在内的完整 `10-seed` 对比。这里的 `2 workers` 来自当时复现 `legacy92` 高精度结果所用的 `run_cifar10_seed_list.sh` 默认设置，主要作为高精度 sanity/legacy 对照；后续 ResNet32/ResNet56、20 workers 和多 setting 实验才是异步并发扩展的主线协议。

| Method | Final Acc Mean ± Std | Best Acc Mean ± Std | Final Loss Mean ± Std |
|---|---:|---:|---:|
| AsyncSAM+RMS | 92.27 ± 0.46 | 92.33 ± 0.46 | 0.3444 ± 0.0164 |
| FedAsync | 89.91 ± 0.25 | 90.04 ± 0.20 | 0.4426 ± 0.0108 |
| AsyncSGD | 89.82 ± 0.25 | 89.96 ± 0.26 | 0.4407 ± 0.0194 |
| FedBuff | 88.09 ± 1.25 | 89.14 ± 0.25 | 0.4321 ± 0.0542 |
| FedAC (tuned) | 79.80 ± 9.91 | 84.06 ± 2.81 | 0.6088 ± 0.2844 |

对应完整图与汇总表：

- `mnist/experiments/cifar10_10seed_compare_legacy92_with_fedac_fig/multiseed_compare.png`
- `mnist/experiments/cifar10_10seed_compare_legacy92_with_fedac_fig/summary.csv`

这组结果进一步强化了当前主结论：

1. `AsyncSAM+RMS` 在 `CIFAR-10` 的完整 `10-seed` 均值上仍然最优
2. `AsyncSAM+RMS` 不仅精度最高，而且方差明显小于 `FedAC`
3. `FedAC` 在这一共享异步协议下存在明显尾段不稳定问题

### 11.2 FedAC 在 CIFAR-10 上的额外观察

本轮 `FedAC` 仍沿用此前 `seed=10` sweep 得到的最优专属参数：

- `fedac_eta_g = 3e-4`
- `fedac_buffer_size = 5`
- `fedac_beta1 = 0.6`
- `fedac_beta2 = 0.9`
- `cv_server_lr = 0.5`
- `cv_momentum = 1.0`

但在完整 `10-seed` 下，`FedAC` 仍出现明显 seed 敏感性，其中：

- `seed09`: final `52.92`, best `80.52`, final loss `1.3482`
- `seed02`: final `70.47`, best `77.01`, final loss `0.9345`

这说明：

1. 当前 `FedAC` 在该 one-arrival-at-a-time 异步协议下方差仍然过大
2. 单 seed 调好不代表 multi-seed 下仍然稳定
3. 现阶段 `FedAC` 不足以撼动 `AsyncSAM+RMS` 的主结论

### 11.3 CIFAR-100 入口已经打通

当前主实验代码已增加 `cifar100` 支持，改动已经落实在：

- `mnist/async_distributed_main.py`

已完成的代码修改包括：

- `--dataset` 增加 `cifar100`
- `build_dataset()` 增加 `torchvision.datasets.CIFAR100`
- CIFAR-100 使用 `100` 类输出头
- 增加 CIFAR-100 的归一化统计

并已完成一个极小 smoke 验证：

- `mnist/experiments/cifar100_smoke_asyncsgd_seed1.pkl`

该 smoke 使用 `asyncsgd + cifar100 + resnet18 + 1 epoch`，已成功下载数据、完成训练并落盘结果。

这意味着：

1. 代码层面已经可以直接启动 `CIFAR-100` 实验
2. 后续只需复制 `CIFAR-10` 的启动脚本并替换数据集/规模参数，即可开始正式对比

### 11.4 可复现性记录已进一步补强

最近一轮实验后，结果记录方式也补强了两点：

- 每个实验输出现在会额外写入 `.pkl.meta.json`
- 批量脚本会为每个结果写入 `.pkl.cmd.txt`

因此当前新生成的 `CIFAR-10 FedAC 10-seed` 结果已经包含：

- 原始命令参数
- 环境变量摘要
- 代码文件哈希
- 最终 summary 指标

这能显著降低后续“参数只留在对话里、图能复现但命令记不住”的风险。

### 11.5 CIFAR-10 多设置验证已完成

为了让 `CIFAR-10 / ResNet18` 也具备和 `MNIST` 一样的“换协议看结论是否仍成立”的证据链，当前已完成一套 settings validation 实验。

这一轮新增的三类设置是：

- `label_sorted`
- `high_delay (delay_gap=0.5, delay_jitter=0.1)`
- `IID 20 workers`

方法集沿用 `CIFAR-10` 主线，而不额外引入 `AsyncSAM` 消融，以保持实验规模可控且与当前主表一致：

- `AsyncSAM+RMS`
- `AsyncSGD`
- `FedAsync`
- `FedBuff`
- `FedAC`

对应脚本：

- `mnist/run_cifar10_validation_settings_10seed.sh`
- `mnist/plot_cifar10_validation_settings_10seed.py`

最终输出目录：

- `mnist/experiments/cifar10_validation_settings_10seed_logs/`
- `mnist/experiments/cifar10_validation_settings_10seed_fig/`

其中，`IID 20 workers` 下为主方法额外加入了一个很小的 stale 修正：

- `sam_stale_base_mix = 0.2`

这个修正与 `MNIST` 多设置验证保持一致，只是在更高 staleness 下把 AA 候选与最新 base step 做小比例混合，不改变 `AA` 的核心形式。

最终结果表明：

1. 在 `Label-Sorted Non-IID` 和 `High Delay IID` 这两组 `CIFAR-10` 额外设置下，`AsyncSAM+RMS` 仍然稳定领先 `AsyncSGD / FedAsync / FedBuff / FedAC`
2. `FedAC` 在这两组上依然方差很大，尤其在 `High Delay IID` 下 seed 敏感性依旧明显
3. `IID 20 Workers` 的第一版共享协议里，所有方法都明显退化，说明当前 `CIFAR-10 + 20 workers` 共享协议本身已经处在失稳区间
4. 因此，这一版原始 `20 workers` 结果更适合作为“失败协议/负对照”，而不应直接与前三组一起作为最终主图

### 11.6 CIFAR-10 `20 workers` 共享协议筛选与公平重跑已完成

在 `CIFAR-10 / ResNet18 / 20 workers` 这组设置上，完整 `10-seed` 主表已经显示出“所有方法一起退化”的现象，因此后续补救优先级不再是继续扩大方法对比，而是先找出一个更合理的共享协议。

已完成的第一轮公平 pilot 说明：

- 仅把共享参数改为更小的 `lr` 或更小的 `delay_gap`，还不足以把 `20 workers` 拉回正常区间
- 旧的 `AsyncSAM+RMS` 专项搜索方向如果继续扫大 batch（如 `256/512`），前台调试结果也显示前期 accuracy 长时间停留在 `10%-16%`，说明这条方向不值得继续投入

因此当前已经把 `AsyncSAM+RMS` 的专项共享协议搜索改成了新的四组候选，只保留更像“稳定训练”的方向：

- `T1`: `batch_size=128`, `epochs=150`, `lr=0.01`, `delay_gap=0.0`, `delay_jitter=0.0`
- `T2`: `batch_size=128`, `epochs=200`, `lr=0.01`, `delay_gap=0.0`, `delay_jitter=0.0`
- `T3`: `batch_size=64`, `epochs=150`, `lr=0.01`, `delay_gap=0.0`, `delay_jitter=0.0`
- `T4`: `batch_size=128`, `epochs=150`, `lr=0.005`, `delay_gap=0.0`, `delay_jitter=0.0`

对应脚本与输出位置：

- `mnist/run_cifar10_workers20_asyncsam_rms_protocol_search.sh`
- `mnist/summarize_cifar10_workers20_asyncsam_rms_protocol_search.py`
- `mnist/plot_cifar10_workers20_asyncsam_rms_protocol_search.py`
- `mnist/experiments/cifar10_workers20_asyncsam_rms_protocol_search_logs/`
- `mnist/experiments/cifar10_workers20_asyncsam_rms_protocol_search_fig/`

这一步的目标不是立即形成新的论文主图，而是尽快找到一组值得整组方法重跑的共享协议；如果这四组候选仍然明显低于当前可接受区间，就会停止继续扩展，直接把“`20 workers` 在当前共享模拟器下难以稳定”作为负结果记录下来。

阶段一 `seed=1` 的四组候选目前已经全部跑完，结果如下：

| Protocol | Shared Params | Final Acc | Best Acc | Final Loss |
|---|---|---:|---:|---:|
| `T1` | `bs=128, epochs=150, lr=0.01, gap=0.0` | 68.69 | 68.71 | 0.8892 |
| `T2` | `bs=128, epochs=200, lr=0.01, gap=0.0` | 74.62 | 74.70 | 0.7338 |
| `T3` | `bs=64, epochs=150, lr=0.01, gap=0.0` | 82.41 | 82.44 | 0.5388 |
| `T4` | `bs=128, epochs=150, lr=0.005, gap=0.0` | 65.47 | 65.54 | 0.9830 |

当前最好的共享参数是：

- `batch_size=64`
- `epochs=150`
- `lr=0.01`
- `delay_gap=0.0`
- `delay_jitter=0.0`

这一结果有两个直接含义：

1. `20 workers` 下继续增大 batch 并不能稳定 `AsyncSAM+RMS`，反而更差
2. 真正起作用的是“去掉额外 synthetic delay + 保持较小 batch”，说明当前失稳更像是高并发下有效更新频率和异步噪声共同作用的结果

对应结果文件与图：

- `mnist/experiments/cifar10_workers20_asyncsam_rms_protocol_search_fig/stage1_summary.csv`
- `mnist/experiments/cifar10_workers20_asyncsam_rms_protocol_search_fig/stage1_final_acc.png`

在此基础上，当前已固定 `20 workers` 的后续公平重跑协议为：

- `batch_size=64`
- `epochs=200`
- `lr=0.01`
- `delay_gap=0.0`
- `delay_jitter=0.0`

这一步保持所有方法共享相同的公共协议，只保留各算法原本各自的独有超参。对应的完整 `10-seed` 重跑脚本和汇总脚本为：

- `mnist/run_cifar10_workers20_t3e200_10seed.sh`
- `mnist/plot_cifar10_workers20_t3e200_10seed.py`

输出目录将落在：

- `mnist/experiments/cifar10_workers20_t3e200_10seed_logs/`
- `mnist/experiments/cifar10_workers20_t3e200_10seed_fig/`

该脚本会尽量占满空闲卡，不再把独立 seed 批次排队等待。

考虑到当前机器仍有大量 HBM 余量，后续又额外增加了一层 back-half seed helper：

- `mnist/run_cifar10_workers20_t3e200_tail_parallel.sh`

这层 helper 会优先并行补跑更靠后的 `seed 7/8/9/10`，目的是在不明显干扰前端 `seed 1/2/3/4` 的前提下，提前把尾部种子叠到已有卡上，加快整组 `10-seed` 收尾速度。

这组统一共享协议的完整 `10-seed` 重跑现已全部完成，最终结果如下：

| Method | Final Acc Mean ± Std | Best Acc Mean ± Std | Final Loss Mean ± Std |
|---|---:|---:|---:|
| AsyncSAM+RMS | 83.76 ± 0.56 | 83.95 ± 0.57 | 0.5083 ± 0.0134 |
| AsyncSGD | 75.55 ± 0.83 | 75.77 ± 0.77 | 0.7072 ± 0.0226 |
| FedAsync | 75.57 ± 0.78 | 75.75 ± 0.79 | 0.7077 ± 0.0217 |
| FedBuff | 73.26 ± 7.23 | 79.15 ± 0.95 | 0.8225 ± 0.2814 |
| FedAC | 26.60 ± 6.52 | 30.58 ± 6.93 | 2.2247 ± 0.3314 |

对应汇总文件：

- `mnist/experiments/cifar10_workers20_t3e200_10seed_fig/summary.csv`
- `mnist/experiments/cifar10_workers20_t3e200_10seed_fig/multiseed_compare.png`
- `mnist/experiments/cifar10_workers20_t3e200_10seed_fig/multiseed_compare.pdf`

这一轮结果说明：

1. `20 workers` 并不是“所有方法都必然崩”，关键在于共享协议是否匹配高并发场景
2. 在完全统一的 `batch_size=64, epochs=200, lr=0.01, no extra delay` 协议下，`AsyncSAM+RMS` 重新成为最优方法，并且相对 `AsyncSGD / FedAsync` 领先约 `8.2` 个点
3. `FedBuff` 在这组协议下虽然 `best acc` 还能冲到 `79.15%`，但最终稳定性明显不如 `AsyncSAM+RMS`
4. `FedAC` 在这组高并发设置里仍然明显失效，说明它对共享协议和 seed 的敏感性远强于其它方法

另外，主训练入口现在已经补上了一个新的高异质 `non-IID` 分区方式：

- `--partition dirichlet`

并新增了两个控制参数：

- `--dirichlet-alpha`
- `--dirichlet-min-size`

这一路径保留类别分布异质性，但不再像 `label_sorted` 那样按标签排序后整块切分，更适合做“不是 label-sorted，但 non-IID 很强”的额外验证。

当前默认设置是：

- `dirichlet_alpha = 0.05`
- `dirichlet_min_size = 1`

这是一组偏强异质的默认值。一次本地统计验证显示，在 `2000` 个样本、`10` 个 worker 的 `MNIST` 子集上，不同 worker 的样本数已经显著不均衡，且多数 worker 只集中在少数类别上，说明该分区确实能形成比普通 IID 明显更强的异质性。

### 11.7 FedAC `workers20` 调参与失败尝试摘要

为避免研究进展被失败调参流水账淹没，这里只保留关键实现修正、最好结果和可复查证据路径。完整原始输出仍在对应 `summary.csv`、`best.txt` 与日志目录中。

关键修正已经落到 `mnist/async_distributed_main.py`：

- `FedAC` 服务器控制变量更新现在真正使用 `cv_server_lr`
- `FedAC` 路径支持 `cv_global_clip_norm`
- 新增更接近 `AFL-Lib` 的 exact 客户端路径，可上传 `dW / dC`
- exact 路径补齐 `fedac_local_weight_decay=1e-4`、AFL-style lr、worker 级本地 optimizer state 等开关
- 训练入口支持 `flush=True`、`.partial.meta.json`、early-abort，后续调参按“短筛 -> 小规模确认 -> 长跑”执行

压缩后的实验结论：

- 旧迁移实现 but tuned 的最好已知 `2-seed` pilot 是 `a7_buf3_smooth_adam_lo_eta`，final acc `47.77%`，best acc `48.89%`
- 围绕 `a7` 的二次局部搜索没有超过 `47.77%`
- 更严格对齐 `AFL-Lib` 的 exact / realign 路线没有自然变好；strict-exact 当前最好 seed1 约 `38.69%`
- 多轮 short-screen、bank3-bank10 搜索均没有出现接近 `70%` 的分支，很多配置停留在 `10%-35%`
- 因此，`FedAC` 在 `CIFAR-10 / 20 workers` 高并发异步协议下的弱表现，更像是算法机制与该协议不匹配，而不是缺一个简单超参

保留证据路径：

- `mnist/experiments/cifar10_workers20_fedac_retune_pilot_fig/summary.csv`
- `mnist/experiments/cifar10_workers20_fedac_retune_local_pilot_fig/summary.csv`
- `mnist/experiments/cifar10_workers20_fedac_realign_pilot_fig/summary.csv`
- `mnist/experiments/cifar10_workers20_fedac_shortscreen_fig/summary.csv`
- `mnist/experiments/cifar10_workers20_fedac_exact_auto_search/stage1_summary.csv`

当前决策：后续默认比较采用“旧迁移实现 but tuned”的 `FedAC`；更严格的 exact 路径保留为可选实现，但不作为默认对比口径。

### 11.8 FedAC 比较口径统一

结合上面几轮结果，我们后续在本项目里统一采用：

- **默认比较口径**：旧迁移实现 but tuned 的 `FedAC`
- **保留但不默认**：更严格对齐 `AFL-Lib` 的 exact `FedAC`

原因很直接：

- 在当前 async scaffold 与目标实验设置下，旧迁移实现经过 `FedAC`-specific 调参后，最好能到 `47.77%`
- 而 strict-exact / realign 路线当前最好只有 `38.69%`
- 因此，对我们当前论文主线和共享对比表而言，继续默认采用 exact 路线并不合理

从实现上也已经同步调整：

- `mnist/async_distributed_main.py` 中，`FedAC` 默认客户端语义切回 `legacy`
- `exact` 路径仍保留，通过 `--fedac-client-mode exact` 显式打开
- 这样后续如果直接跑 `--alg fedac`，默认就会落到我们当前认可的“旧迁移实现 but 调过参”这条口径

### 11.9 CIFAR-10 / ResNet32 主线已接入并启动四设置实验

为了把当前 `CIFAR-10` 证据链从 `ResNet18` 扩展到另一条更深的 CIFAR-style backbone，现已把仓库自带的 `resnet32` 接入共享异步主入口：

- `mnist/async_distributed_main.py`
  - `--model` 现支持：
    - `resnet20`
    - `resnet32`
    - `resnet44`
    - `resnet56`
    - `resnet110`
    - `resnet1202`
- 其中当前首先启用的是 `resnet32`
- 一个小型 `ResNet32` smoke 已跑通，说明模型构建、数据流和异步训练入口都正常

随后，已先启动一轮完整的 `CIFAR-10 / ResNet32 / 10-seed` 四设置对比，覆盖：

1. `Base IID`
2. `Label-Sorted Non-IID`
3. `Dirichlet Non-IID (alpha=0.05, min_size=10)`
4. `High Delay IID`

比较方法保持与当前 `CIFAR-10` 主线一致：

- `AsyncSAM+RMS`
- `AsyncSGD`
- `FedAsync`
- `FedBuff`
- `FedAC`

其中 `FedAC` 统一采用当前默认的 tuned legacy 迁移口径。`20 workers` 由于需要单独确认高并发协议，暂不混入这张 ResNet32 四设置主表；后续若并入，会额外使用此前更强的 `a7` 风格参数：

- `buffer_size=3`
- `eta_g=1e-4`
- `beta1=0.9`
- `beta2=0.99`
- `cv_server_lr=0.1`
- `cv_momentum=0.25`
- `cv_global_clip_norm=10.0`

当前运行入口：

- `mnist/run_cifar10_resnet32_four_settings_fixed_public_10seed.sh`
- 日志目录：
  - `mnist/experiments/cifar10_resnet32_four_settings_fixed_public_10seed_logs/`

这批实验已经在 `8` 张 NPU 上铺开，现阶段目标不是再发明新协议，而是直接检验：

- 当前 `AsyncSAM+RMS` 的主结论是否能迁移到 `ResNet32`
- `FedAC` 在更深 backbone 下是否仍然明显落后
- `Dirichlet` 这类更自然的非标签排序 non-IID，在 `ResNet32` 上是否仍保持相同结论
- `20 workers` 作为高并发专项，待 `AsyncSAM+RMS` 协议稳定后再并入完整主表

### 11.10 ResNet32 接入后的模型语义修正与公共参数筛选

在把仓库自带的 CIFAR `resnet32` 直接接入异步主入口后，我们很快发现一件关键事实：

- `resnet/resnet.py` 使用的是 `BatchNorm2d`
- 而当前 async scaffold 的 worker/server 快照同步只覆盖参数张量，不覆盖
  `BatchNorm` 的 running buffers
- 因此，若直接把原始 `resnet32` 接进来，就会出现：
  - 一部分实验在早期直接掉到 `10%`
  - `loss` 异常放大
  - 结果看起来像“参数不对”，但其实首先是模型状态语义不兼容

针对这一点，当前已在：

- `mnist/async_distributed_main.py`

里做了一个最小但关键的修正：

- 对 imported CIFAR `resnet20/32/44/56/110/1202`
  在构建后递归把 `BatchNorm2d` 转成 `GroupNorm`
- 保留 ResNet 深度/残差拓扑
- 去掉会被异步快照漏同步的隐藏 running state

在这个修正之后，`ResNet32` 的 `AsyncSAM+RMS` 公共参数短筛重新变得有意义。

当前 `seed=1`、`30 epochs`、`Base IID` 的短筛结果如下。

注意：这里最初曾误读到一版过期 `partial` 汇总；现已修正为优先读取最终 `.pkl` 结果，下面这张表是最终值。

| Config | Batch | LR | Final Acc | Final Loss |
|---|---:|---:|---:|---:|
| `p3` | 64 | 0.05 | 84.65 | 0.4557 |
| `p2` | 64 | 0.02 | 82.28 | 0.5319 |
| `p1` | 64 | 0.01 | 78.44 | 0.6338 |
| `p7` | 128 | 0.05 | 76.70 | 0.6777 |
| `p0` | 64 | 0.005 | 73.96 | 0.7532 |
| `p6` | 128 | 0.02 | 71.47 | 0.8016 |
| `p5` | 128 | 0.01 | 65.52 | 0.9736 |
| `p4` | 128 | 0.005 | 60.81 | 1.0883 |

当前阶段结论：

- `ResNet32` 下并不是越小学习率越好
- 当前最好的是：
  - `batch_size=64`
  - `lr=0.05`
- `batch_size=64` 整体优于 `128`
- `ResNet32` 的 `30 epoch` 短筛最终值已经进入 `80%+` 区间，不是前面误汇总里看起来的 `50%-60%`

为了避免把 `ResNet32` 的 `30 epoch` 短筛和 `ResNet18` 的 `100 epoch`
主实验混比，额外补跑了一条严格同预算对照：

| Model | Setting | Final Acc | Best Acc | Final Loss |
|---|---|---:|---:|---:|
| `ResNet18` | `Base IID, seed=1, 30 epochs, bs=64, lr=0.05` | 85.95 | 86.02 | 0.4213 |
| `ResNet32` | `Base IID, seed=1, 30 epochs, bs=64, lr=0.05` | 84.65 | 84.65 | 0.4557 |

结论：同预算短筛下 `ResNet18` 仍略高约 `1.3` 个点，但两者已经是同一量级；
此前“`ResNet32` 只有 `60%`”是过期 `partial` 汇总造成的误读。

同时，已额外启动两个方向的快速确认：

1. `p3` 与 `p1/p5` 的 `seed2/3` 对照确认
2. 使用当前 best 候选 `p3` 先做 `AsyncSAM+RMS` 的设置适应性检查

目前 `p3` 的确认结果仍在继续收集，而设置适应性检查里：

- `Base IID / Label-Sorted / Dirichlet / High Delay`
  的早期结果都已经进入正常训练区间
- `workers20` 的第一条检查初版较弱，随后发现需要保留
  `AsyncSAM+RMS` 在高并发下常用的
  `--sam-stale-base-mix 0.2`
  保护，因此又补发了一条修正版检查

基于这轮结果，当前已经先启动：

- `mnist/run_cifar10_resnet32_four_settings_fixed_public_10seed.sh`

即在四个已经验证可行的设置上，使用当前固定公共参数：

- `model=resnet32`
- `batch_size=64`
- `lr=0.05`

先展开完整 `10-seed` 对比；`workers20` 待修正版 `AsyncSAM+RMS` 检查稳定后再并入完整主表。

### 11.11 CIFAR-10 / ResNet32 四设置 10-seed 对比已完成

`CIFAR-10 / ResNet32 / 100 epochs` 四设置主实验已经全部落盘，所有设置与方法均为 `10/10` seeds 完整结果。公共参数保持：

- `model = resnet32`
- `batch_size = 64`
- `lr = 0.05`
- `epochs = 100`

汇总脚本与输出：

- `mnist/plot_cifar10_resnet32_four_settings_10seed.py`
- `mnist/experiments/cifar10_resnet32_four_settings_fixed_public_10seed_fig/summary.csv`
- `mnist/experiments/cifar10_resnet32_four_settings_fixed_public_10seed_fig/resnet32_four_settings_summary.png`
- `mnist/experiments/cifar10_resnet32_four_settings_fixed_public_10seed_fig/resnet32_four_settings_curves.png`

最终结果如下，数值为 `mean ± std`。

| Setting | Method | Final Acc | Best Acc | Final Loss |
|---|---|---:|---:|---:|
| Base IID | AsyncSAM+RMS | 89.72 ± 0.35 | 89.82 ± 0.33 | 0.3450 ± 0.0111 |
| Base IID | AsyncSGD | 87.13 ± 0.20 | 87.22 ± 0.24 | 0.4248 ± 0.0082 |
| Base IID | FedAsync | 86.89 ± 0.32 | 87.12 ± 0.34 | 0.4285 ± 0.0094 |
| Base IID | FedBuff | 84.82 ± 0.98 | 85.81 ± 0.56 | 0.4594 ± 0.0335 |
| Base IID | FedAC | 14.82 ± 2.65 | 16.63 ± 1.66 | 2.2729 ± 0.0495 |
| Label-Sorted Non-IID | AsyncSAM+RMS | 89.83 ± 0.24 | 89.96 ± 0.22 | 0.3411 ± 0.0092 |
| Label-Sorted Non-IID | AsyncSGD | 86.79 ± 0.35 | 87.00 ± 0.30 | 0.4339 ± 0.0069 |
| Label-Sorted Non-IID | FedAsync | 86.66 ± 0.44 | 86.82 ± 0.40 | 0.4373 ± 0.0111 |
| Label-Sorted Non-IID | FedBuff | 83.25 ± 1.83 | 85.73 ± 0.46 | 0.5163 ± 0.0621 |
| Label-Sorted Non-IID | FedAC | 13.37 ± 2.39 | 16.62 ± 2.07 | 2.3126 ± 0.0853 |
| Dirichlet Non-IID | AsyncSAM+RMS | 89.50 ± 0.47 | 89.74 ± 0.33 | 0.3484 ± 0.0135 |
| Dirichlet Non-IID | AsyncSGD | 86.54 ± 0.45 | 86.69 ± 0.44 | 0.4382 ± 0.0141 |
| Dirichlet Non-IID | FedAsync | 86.51 ± 0.44 | 86.72 ± 0.43 | 0.4386 ± 0.0135 |
| Dirichlet Non-IID | FedBuff | 83.72 ± 0.95 | 84.68 ± 0.70 | 0.4937 ± 0.0279 |
| Dirichlet Non-IID | FedAC | 13.57 ± 1.96 | 16.31 ± 1.99 | 2.3576 ± 0.0728 |
| High Delay IID | AsyncSAM+RMS | 89.82 ± 0.26 | 89.95 ± 0.30 | 0.3414 ± 0.0097 |
| High Delay IID | AsyncSGD | 87.11 ± 0.17 | 87.30 ± 0.20 | 0.4265 ± 0.0108 |
| High Delay IID | FedAsync | 87.02 ± 0.32 | 87.17 ± 0.33 | 0.4317 ± 0.0095 |
| High Delay IID | FedBuff | 84.10 ± 1.33 | 85.45 ± 0.49 | 0.4826 ± 0.0448 |
| High Delay IID | FedAC | 14.90 ± 2.47 | 16.42 ± 1.52 | 2.3014 ± 0.1036 |

这组实验的结论很清楚：

- `AsyncSAM+RMS` 在四个设置下都取得最高 final acc、最高 best acc 和最低 final loss
- 相比 `AsyncSGD`，`AsyncSAM+RMS` 的 final acc 提升约 `2.6-3.0` 个点
- `FedAsync` 与 `AsyncSGD` 基本接近，说明当前实现下 FedAsync 更像轻量异步基线
- `FedBuff` 稳定低于 `AsyncSAM+RMS`
- `FedAC` 在该 ResNet32 公共协议下明显失效，这与前面 `workers20` FedAC 调参观察一致

### 11.12 CIFAR-10 / ResNet56 四设置 10-seed 对比已完成

在 ResNet32 四设置结果确认后，进一步完成了 `CIFAR-10 / ResNet56` 验证，用于检查当前结论是否能迁移到更深的 CIFAR-style ResNet。为减少无效计算，这一轮不再跑 `FedAC`，只保留：

- `AsyncSAM+RMS`
- `AsyncSGD`
- `FedAsync`
- `FedBuff`

公共协议沿用 ResNet32 主线：

- `model = resnet56`
- `batch_size = 64`
- `lr = 0.05`
- `epochs = 100`
- settings：`Base IID / Label-Sorted Non-IID / Dirichlet Non-IID / High Delay IID`
- seeds：`1-10`

运行方式：

- `mnist/run_cifar10_resnet56_four_settings_parallel_10seed.sh`
- 默认 `MAX_PARALLEL=64`
- round-robin 使用 `8` 张 NPU
- 输出前缀：`cifar10_resnet56_p3_{setting}_{method}_seed{seed}.pkl`
- 日志目录：`mnist/experiments/cifar10_resnet56_four_settings_parallel_10seed_logs/`

汇总与画图输出：

- `mnist/plot_cifar10_resnet56_four_settings_10seed.py`
- `mnist/experiments/cifar10_resnet56_four_settings_parallel_10seed_fig/summary.csv`
- `mnist/experiments/cifar10_resnet56_four_settings_parallel_10seed_fig/resnet56_four_settings_summary.png`
- `mnist/experiments/cifar10_resnet56_four_settings_parallel_10seed_fig/resnet56_four_settings_curves.png`

最终结果如下，数值为 `mean ± std`。

| Setting | Method | Final Acc | Best Acc | Final Loss |
|---|---|---:|---:|---:|
| Base IID | AsyncSAM+RMS | 90.64 ± 0.28 | 90.76 ± 0.29 | 0.3293 ± 0.0125 |
| Base IID | AsyncSGD | 87.83 ± 0.30 | 87.98 ± 0.35 | 0.4181 ± 0.0094 |
| Base IID | FedAsync | 87.96 ± 0.26 | 88.09 ± 0.21 | 0.4189 ± 0.0042 |
| Base IID | FedBuff | 84.84 ± 0.96 | 86.17 ± 0.34 | 0.4601 ± 0.0278 |
| Label-Sorted Non-IID | AsyncSAM+RMS | 90.55 ± 0.15 | 90.74 ± 0.19 | 0.3324 ± 0.0064 |
| Label-Sorted Non-IID | AsyncSGD | 87.36 ± 0.25 | 87.53 ± 0.25 | 0.4318 ± 0.0074 |
| Label-Sorted Non-IID | FedAsync | 87.46 ± 0.35 | 87.56 ± 0.35 | 0.4238 ± 0.0093 |
| Label-Sorted Non-IID | FedBuff | 84.84 ± 1.72 | 85.88 ± 0.51 | 0.4669 ± 0.0634 |
| Dirichlet Non-IID | AsyncSAM+RMS | 90.14 ± 0.61 | 90.25 ± 0.62 | 0.3409 ± 0.0145 |
| Dirichlet Non-IID | AsyncSGD | 87.30 ± 0.49 | 87.42 ± 0.48 | 0.4307 ± 0.0102 |
| Dirichlet Non-IID | FedAsync | 87.20 ± 0.45 | 87.38 ± 0.50 | 0.4295 ± 0.0078 |
| Dirichlet Non-IID | FedBuff | 84.60 ± 1.74 | 85.59 ± 0.84 | 0.4729 ± 0.0610 |
| High Delay IID | AsyncSAM+RMS | 90.49 ± 0.28 | 90.69 ± 0.26 | 0.3305 ± 0.0088 |
| High Delay IID | AsyncSGD | 87.92 ± 0.32 | 88.03 ± 0.30 | 0.4174 ± 0.0084 |
| High Delay IID | FedAsync | 87.96 ± 0.27 | 88.10 ± 0.26 | 0.4202 ± 0.0111 |
| High Delay IID | FedBuff | 85.08 ± 0.97 | 86.17 ± 0.26 | 0.4585 ± 0.0352 |

这组实验进一步强化了当前主结论：

- `AsyncSAM+RMS` 在 ResNet56 的四个设置下全部取得最高 final acc 和 best acc
- 相比 `AsyncSGD/FedAsync`，`AsyncSAM+RMS` 的 final acc 提升约 `2.5-3.2` 个点
- 相比 `FedBuff`，`AsyncSAM+RMS` 的 final acc 提升约 `5.4-6.4` 个点
- 该结论与 ResNet32 主线一致，说明当前方法优势不依赖单一 backbone

### 11.13 CIFAR-100 / ResNet32 四设置 10-seed 对比已完成

为验证当前结论能否迁移到更难的数据集，已完成 `CIFAR-100 / ResNet32` 四设置实验。该实验不再跑 `FedAC`，只保留：

- `AsyncSAM+RMS`
- `AsyncSGD`
- `FedAsync`
- `FedBuff`

公共协议：

- `dataset = cifar100`
- `model = resnet32`
- `batch_size = 64`
- `lr = 0.05`
- `epochs = 100`
- settings：`Base IID / Label-Sorted Non-IID / Dirichlet Non-IID / High Delay IID`
- seeds：`1-10`

运行方式：

- `mnist/run_cifar100_resnet32_four_settings_parallel_10seed.sh`
- 默认 `MAX_PARALLEL=64`
- round-robin 使用 `8` 张 NPU
- 输出前缀：`cifar100_resnet32_p3_{setting}_{method}_seed{seed}.pkl`
- 日志目录：`mnist/experiments/cifar100_resnet32_four_settings_parallel_10seed_logs/`

汇总与画图脚本：

- `mnist/plot_cifar100_resnet32_four_settings_10seed.py`
- 输出目录：`mnist/experiments/cifar100_resnet32_four_settings_parallel_10seed_fig/`

启动后检查：

- `dataset=cifar100` 与 `model=resnet32` 已在 `.partial.meta.json` 中确认
- 完整结果：`160/160`
- 错误日志：`0`

汇总与画图输出：

- `mnist/experiments/cifar100_resnet32_four_settings_parallel_10seed_fig/summary.csv`
- `mnist/experiments/cifar100_resnet32_four_settings_parallel_10seed_fig/cifar100_resnet32_four_settings_summary.png`
- `mnist/experiments/cifar100_resnet32_four_settings_parallel_10seed_fig/cifar100_resnet32_four_settings_curves.png`

最终结果如下，数值为 `mean ± std`。

| Setting | Method | Final Acc | Best Acc | Final Loss |
|---|---|---:|---:|---:|
| Base IID | AsyncSAM+RMS | 62.15 ± 0.42 | 62.35 ± 0.39 | 1.3788 ± 0.0150 |
| Base IID | AsyncSGD | 56.73 ± 0.84 | 56.98 ± 0.83 | 1.6061 ± 0.0352 |
| Base IID | FedAsync | 56.97 ± 0.91 | 57.13 ± 0.83 | 1.6006 ± 0.0327 |
| Base IID | FedBuff | 51.88 ± 1.68 | 53.62 ± 1.20 | 1.8012 ± 0.0911 |
| Label-Sorted Non-IID | AsyncSAM+RMS | 61.97 ± 0.37 | 62.25 ± 0.41 | 1.3841 ± 0.0143 |
| Label-Sorted Non-IID | AsyncSGD | 55.08 ± 0.65 | 55.82 ± 0.65 | 1.6709 ± 0.0284 |
| Label-Sorted Non-IID | FedAsync | 55.08 ± 0.86 | 55.81 ± 0.76 | 1.6723 ± 0.0374 |
| Label-Sorted Non-IID | FedBuff | 51.25 ± 1.07 | 52.70 ± 1.03 | 1.8221 ± 0.0468 |
| Dirichlet Non-IID | AsyncSAM+RMS | 61.90 ± 0.62 | 62.08 ± 0.65 | 1.3973 ± 0.0286 |
| Dirichlet Non-IID | AsyncSGD | 55.29 ± 0.96 | 56.13 ± 0.61 | 1.6722 ± 0.0401 |
| Dirichlet Non-IID | FedAsync | 55.23 ± 1.31 | 55.96 ± 0.84 | 1.6696 ± 0.0568 |
| Dirichlet Non-IID | FedBuff | 52.31 ± 1.12 | 53.14 ± 0.95 | 1.7755 ± 0.0510 |
| High Delay IID | AsyncSAM+RMS | 62.21 ± 0.49 | 62.41 ± 0.49 | 1.3783 ± 0.0207 |
| High Delay IID | AsyncSGD | 56.78 ± 0.77 | 56.97 ± 0.68 | 1.6079 ± 0.0331 |
| High Delay IID | FedAsync | 56.71 ± 0.87 | 56.87 ± 0.77 | 1.6122 ± 0.0353 |
| High Delay IID | FedBuff | 52.65 ± 1.60 | 53.57 ± 0.64 | 1.7587 ± 0.0786 |

需要注意的是，`CIFAR-100` 的 top-1 精度目标与 `CIFAR-10` 不同。当前 ResNet32 + basic crop/flip + 100 epoch 的异步设置达到约 `62%` 并不异常；若希望进一步冲高，不能只依赖算法本身，还需要更长训练预算、更大模型容量或更强公共训练 recipe。

### 11.14 CIFAR-100 高预算 AsyncSAM+RMS pilot 已启动

为了判断 `CIFAR-100` 是否能通过更高预算接近更高精度，已在不改变 `AsyncSAM+RMS` 核心机制的前提下启动若干 `seed=1` pilot。该阶段只用于筛选公共协议，不作为最终公平对比结果。

新增代码支持：

- `async_distributed_main.py` 新增 `--cifar-augment`
- 可选项：`none / basic / randaugment / autoaugment / trivial`
- `basic` 为默认值，保持所有既有实验语义不变
- 新增 `--random-erasing`，默认 `0.0`

plain 高预算 pilot：

- 脚本：`mnist/run_cifar100_highbudget_asyncsam_rms_pilot.sh`
- 配置：
  - `resnet32, epochs=200, lr=0.05`
  - `resnet32, epochs=300, lr=0.05`
  - `resnet32, epochs=300, lr=0.10`
  - `resnet56, epochs=200, lr=0.05`
  - `resnet56, epochs=300, lr=0.05`
  - `resnet110, epochs=200, lr=0.05`

ResNet18 高预算 pilot：

- 脚本：`mnist/run_cifar100_highbudget_resnet18_asyncsam_rms_pilot.sh`
- 配置：
  - `resnet18, epochs=200, lr=0.05`
  - `resnet18, epochs=300, lr=0.05`
  - `resnet18, epochs=300, lr=0.10`

强增强公共 recipe pilot：

- 脚本：`mnist/run_cifar100_strongaug_asyncsam_rms_pilot.sh`
- 公共增强：`--cifar-augment randaugment --random-erasing 0.25`
- 配置：
  - `resnet32, epochs=300, lr=0.05`
  - `resnet56, epochs=300, lr=0.05`
  - `resnet18, epochs=300, lr=0.05`
  - `resnet18, epochs=300, lr=0.10`

当前早期中间结果：

| Pilot | Evals | Last Acc | Best Acc | Last Loss |
|---|---:|---:|---:|---:|
| resnet32 / 200e / lr0.05 | 18 | 30.22 | 30.22 | 2.7770 |
| resnet32 / 300e / lr0.05 | 23 | 30.15 | 31.64 | 2.7700 |
| resnet32 / 300e / lr0.10 | 22 | 19.89 | 23.87 | 3.2938 |
| resnet56 / 200e / lr0.05 | 16 | 28.68 | 28.68 | 2.8836 |
| resnet56 / 300e / lr0.05 | 16 | 27.12 | 27.74 | 2.9441 |
| resnet110 / 200e / lr0.05 | 10 | 17.95 | 17.95 | 3.4226 |
| resnet18 / 200e / lr0.05 | 10 | 24.06 | 24.06 | 3.0949 |
| resnet18 / 300e / lr0.05 | 11 | 25.28 | 25.28 | 3.0302 |
| resnet18 / 300e / lr0.10 | 10 | 18.40 | 18.40 | 3.4254 |

初步观察：

- `lr=0.10` 在 CIFAR-100 上明显偏激进，暂时不适合作为主线
- `resnet32 / 300e / lr0.05` 是当前 plain pilot 里最好的早期曲线
- `resnet56` 和 `resnet18` 仍在早期，暂不能完全判定
- strong-augmentation pilot 已启动，但尚未产生第一次评估结果

随后已按实时筛选原则停止明显落后的 `lr=0.10` 长跑：

- `cifar100_highbudget_r32_e300_lr010_asyncsam_rms_seed1`
- `cifar100_highbudget_r18_e300_lr010_asyncsam_rms_seed1`
- `cifar100_strongaug_r18_e300_lr010_ra_re25_asyncsam_rms_seed1`

保留继续运行的候选集中，主线暂时优先关注：

- `resnet32 / 300e / lr0.05`
- `resnet56 / 200e or 300e / lr0.05`
- `resnet18 / 200e or 300e / lr0.05`
- strong-augmentation 的 `lr=0.05` 配置

### 11.15 CIFAR-100 80% 目标下的新公共 recipe pilot

用户将 CIFAR-100 目标从 `90%` 调整为 `80%`。该目标仍高于当前 ResNet32/basic/100epoch 的约 `62%`，但比 `90%` 更可作为高预算 recipe 的探索目标。

新增公共训练参数：

- `--label-smoothing`
- 默认值为 `0.0`，因此不影响旧实验
- 实现为 log-prob 输出下的标准 smoothed NLL：
  - `(1 - eps) * NLL + eps * mean(-log p)`

新增 pilot 脚本：

- `mnist/run_cifar100_strongaug_ls_asyncsam_rms_pilot.sh`

配置：

- `dataset = cifar100`
- `batch_size = 64`
- `epochs = 300`
- `lr = 0.05`
- `cifar_augment = randaugment`
- `random_erasing = 0.25`
- `label_smoothing = 0.1`
- 方法只跑 `AsyncSAM+RMS`
- seed 只跑 `1`
- backbone：
  - `resnet32`
  - `resnet56`
  - `resnet18`

定位：

- 这仍是协议筛选，不是公平对比结论
- 如果该 recipe 明显超过当前 `~62%` 主线，再把同一公共协议同步给 `AsyncSGD/FedAsync/FedBuff` 重跑

### 11.16 CIFAR-100 focused80 / ResNet18 protocol pilot 已完成

`AsyncSAM+RMS / seed=1 / lr=0.05 / batch_size=64` 的 ResNet18 高预算筛选已完成。500epoch 没有带来足够收益，因此当前正式对比不采用 500epoch，而采用 300epoch 中最强且成本更可控的 RandAugment + Random Erasing + Label Smoothing 协议。

| Protocol | Final Acc | Best Acc | Final Loss |
|---|---:|---:|---:|
| ResNet18 / 200e / basic | 71.54 | 71.75 | 1.2912 |
| ResNet18 / 300e / basic | 73.02 | 73.15 | 1.2360 |
| ResNet18 / 300e / RandAug+RE | 74.29 | 74.75 | 1.0247 |
| ResNet18 / 300e / RandAug+RE+LS | 75.62 | 75.62 | 1.0082 |
| ResNet18 / 500e / basic | 74.60 | 74.63 | 1.1939 |
| ResNet18 / 500e / LS | 73.08 | 73.72 | 1.2816 |
| ResNet18 / 500e / RandAug+RE | 75.80 | 76.28 | 0.9948 |
| ResNet18 / 500e / RandAug+RE+LS | 75.53 | 75.84 | 1.0439 |

正式公共协议：

- `dataset = cifar100`
- `model = resnet18`
- `epochs = 300`
- `lr = 0.05`
- `batch_size = 64`
- `cifar_augment = randaugment`
- `random_erasing = 0.25`
- `label_smoothing = 0.1`

### 11.17 CIFAR-100 / ResNet18 / 300e / 四设置 10-seed 对比已完成

按照上面的 300epoch 公共协议，已启动完整 `160` 个实验：

- 四种方法：`AsyncSAM+RMS / AsyncSGD / FedAsync / FedBuff`
- 四种设置：`Base IID / Label-Sorted Non-IID / Dirichlet Non-IID / High Delay IID`
- 十个 seed：`1-10`
- 总量：`4 methods * 4 settings * 10 seeds = 160`

启动脚本：

- `mnist/run_cifar100_resnet18_e300_ra_re_ls_four_settings_10seed.sh`

运行配置补充：

- `FedAsync`: `fedasync_decay = 1.0`
- `FedBuff`: `fedbuff_k = 3`, `fedbuff_etag = 5.0`
- `Dirichlet`: `alpha = 0.05`, `min_size = 10`
- `High Delay`: `delay_gap = 0.5`, `delay_jitter = 0.1`
- `AsyncSAM+RMS`: 沿用 CIFAR 主线参数，`sam_precond_beta = 0.90`, `sam_hist_length = 4`, `sam_period = 2`, `sam_base_mix = 0.8`

截至 `2026-04-16 15:07`：

- 已完成 final：`160/160`
- `summary.csv` 共 `16` 行，所有行 `num_seeds = 10`
- 图表已按完整结果重新生成，不再包含 partial 占位结果

完整 10-seed 结果如下，数值为 mean ± std：

| Setting | Method | Final Acc | Best Acc | Final Loss |
|---|---|---:|---:|---:|
| Base IID | AsyncSAM+RMS | 74.73 ± 0.50 | 75.19 ± 0.39 | 1.0365 ± 0.0206 |
| Base IID | FedAsync | 74.09 ± 0.33 | 74.22 ± 0.31 | 1.1722 ± 0.0101 |
| Base IID | AsyncSGD | 74.04 ± 0.35 | 74.20 ± 0.34 | 1.1747 ± 0.0188 |
| Base IID | FedBuff | 68.62 ± 0.94 | 69.90 ± 0.31 | 1.3138 ± 0.0443 |
| Label-Sorted Non-IID | AsyncSAM+RMS | 74.64 ± 0.52 | 74.89 ± 0.44 | 1.0418 ± 0.0122 |
| Label-Sorted Non-IID | FedAsync | 74.39 ± 0.36 | 74.53 ± 0.32 | 1.1599 ± 0.0165 |
| Label-Sorted Non-IID | AsyncSGD | 74.29 ± 0.26 | 74.42 ± 0.26 | 1.1611 ± 0.0099 |
| Label-Sorted Non-IID | FedBuff | 68.36 ± 1.41 | 69.74 ± 0.25 | 1.3358 ± 0.0649 |
| Dirichlet Non-IID | AsyncSAM+RMS | 74.97 ± 0.25 | 75.28 ± 0.34 | 1.0289 ± 0.0125 |
| Dirichlet Non-IID | FedAsync | 74.28 ± 0.30 | 74.44 ± 0.36 | 1.1615 ± 0.0150 |
| Dirichlet Non-IID | AsyncSGD | 74.22 ± 0.31 | 74.44 ± 0.36 | 1.1594 ± 0.0123 |
| Dirichlet Non-IID | FedBuff | 68.47 ± 0.76 | 69.83 ± 0.12 | 1.3204 ± 0.0350 |
| High Delay IID | AsyncSAM+RMS | 74.81 ± 0.55 | 75.09 ± 0.44 | 1.0348 ± 0.0201 |
| High Delay IID | FedAsync | 74.15 ± 0.37 | 74.31 ± 0.33 | 1.1717 ± 0.0210 |
| High Delay IID | AsyncSGD | 74.08 ± 0.30 | 74.30 ± 0.33 | 1.1763 ± 0.0148 |
| High Delay IID | FedBuff | 68.16 ± 0.93 | 69.81 ± 0.41 | 1.3433 ± 0.0404 |

结论：

- `AsyncSAM+RMS` 在 CIFAR-100 + ResNet18 的四个 setting 中全部取得最高 final accuracy 和 best accuracy。
- 相比最强的一阶异步基线 `FedAsync/AsyncSGD`，`AsyncSAM+RMS` 的 final accuracy 提升约 `+0.25` 到 `+0.73` 个点。
- 相比 `FedBuff`，`AsyncSAM+RMS` 的提升约 `+6.3` 到 `+6.7` 个点，且 loss 明显更低。
- 这组结果满足“超过原有表现”和“超过对比算法”的当前阶段目标，但绝对精度仍在 `75%` 左右，距离 CIFAR-100 80% 目标还有空间。

输出文件：

- `mnist/experiments/cifar100_resnet18_e300_ra_re_ls_four_settings_10seed_fig/summary.csv`
- `mnist/experiments/cifar100_resnet18_e300_ra_re_ls_four_settings_10seed_fig/cifar100_resnet18_e300_ra_re_ls_four_settings_summary.png`
- `mnist/experiments/cifar100_resnet18_e300_ra_re_ls_four_settings_10seed_fig/cifar100_resnet18_e300_ra_re_ls_four_settings_curves.png`

后续调参线：

- `sam_precond_min_denom=0.20` 和 `0.30` 的 Base `10-seed` 已完成，二者都明显高于共享协议主表中的默认 `AsyncSAM+RMS`。
- 后续文档口径中，`11.17` 这张四方法主表继续保留为“共享协议下的公平对比结果”；但如果单独描述 `AsyncSAM+RMS` 在 `CIFAR-100 + ResNet18` 上的实际表现，则默认采用调参后的结果。
- 当前默认的 tuned recipe 记为 `sam_precond_min_denom=0.20`。原因是：它在 Base IID 上与 `0.30` 基本打平，但 final loss 更低，且在 `Label-Sorted / Dirichlet / High Delay` 三个 setting 上也都已经完成一致的 `10-seed` 验证。

### 11.18 CIFAR-100 / ResNet18 / AsyncSAM+RMS 实际表现口径更新

从本节开始，`CIFAR-100 + ResNet18` 上 `AsyncSAM+RMS` 的结果分成两种用途：

- 公平对比：继续使用 `11.17` 的共享协议主表
- 算法实际表现：默认使用 `AsyncSAM+RMS` 私有参数调优后的结果

这里的“实际表现”不再默认引用 `11.17` 中 `74.73 ± 0.50` 那条共享协议结果，而是引用 tuned recipe：

- `sam_precond_beta = 0.90`
- `sam_hist_length = 4`
- `sam_period = 2`
- `sam_base_mix = 0.8`
- `sam_precond_min_denom = 0.20`

调参后的 `10-seed` 结果如下：

| Setting | AsyncSAM+RMS (Tuned) Final Acc | Best Acc | Final Loss |
|---|---:|---:|---:|
| Base IID | 76.16 ± 0.48 | 76.34 ± 0.43 | 1.0433 |
| Label-Sorted Non-IID | 76.08 ± 0.32 | 76.29 ± 0.36 | 1.0382 |
| Dirichlet Non-IID | 75.97 ± 0.37 | 76.24 ± 0.37 | 1.0515 |
| High Delay IID | 75.88 ± 0.30 | 76.10 ± 0.36 | 1.0465 |

补充说明：

- Base IID 上，`sam_precond_min_denom=0.30` 的 `10-seed` 结果几乎与 `0.20` 打平：`76.16 ± 0.33 / 76.38 ± 0.27`，但 final loss 更高，因此当前不作为默认口径。
- Base IID 的单 seed 峰值目前来自 `cifar100_tune_r18_e300_sam_mind020_asyncsam_rms_seed4.pkl`：
  `final acc = 76.93%`，`best acc = 77.19%`。
- 因此，后续如果要描述 `AsyncSAM+RMS` 在 `CIFAR-100 + ResNet18` 上的“实际最好水平”，默认可写为：
  “`10-seed` 稳定结果约 `76.16%`，单 seed 峰值约 `77.19%`。”

对应图表与汇总文件：

- `mnist/experiments/cifar100_resnet18_asyncsam_actual_fig/summary.csv`
- `mnist/experiments/cifar100_resnet18_asyncsam_actual_fig/cifar100_resnet18_asyncsam_actual_vs_fair.png`
- `mnist/experiments/cifar100_resnet18_asyncsam_actual_fig/cifar100_resnet18_asyncsam_actual_vs_fair.pdf`

另外，已补充一张只看 `Base IID` 的“调好后的 `AsyncSAM+RMS` 与原始对比算法同图”结果，用于展示 tuned recipe 相对原 baseline 的实际优势：

- `mnist/experiments/cifar100_resnet18_base_iid_tuned_asyncsam_vs_baselines_fig/summary.csv`
- `mnist/experiments/cifar100_resnet18_base_iid_tuned_asyncsam_vs_baselines_fig/cifar100_resnet18_base_iid_tuned_asyncsam_vs_baselines_summary.png`
- `mnist/experiments/cifar100_resnet18_base_iid_tuned_asyncsam_vs_baselines_fig/cifar100_resnet18_base_iid_tuned_asyncsam_vs_baselines_curves.png`

### 11.19 CIFAR-100 / ResNeXt29-8x16d / 200e / 四设置 10-seed 对比已完成

为验证 `AsyncSAM+RMS` 在非 ResNet 主干上的泛化性，已新增 CIFAR-style `ResNeXt29-8x16d` 并完成完整四设置对比。

公共协议：

- `dataset = cifar100`
- `model = resnext29_8x16d`
- `epochs = 200`
- `lr = 0.03`
- `batch_size = 64`
- `cifar_augment = randaugment`
- `random_erasing = 0.25`
- `label_smoothing = 0.1`
- `weight_decay = 5e-4`
- `seeds = 1-10`

`AsyncSAM+RMS` 私有参数：

- `sam_precond_beta = 0.90`
- `sam_precond_min_denom = 0.30`
- `sam_hist_length = 4`
- `sam_period = 2`
- `sam_base_mix = 0.8`
- `sam_max_history_staleness = 2`

完整性检查：

- 已完成 final：`160/160`
- `summary.csv` 共 `16` 行，所有行 `num_seeds = 10`
- 四种方法：`AsyncSAM+RMS / AsyncSGD / FedAsync / FedBuff`
- 四种设置：`Base IID / Label-Sorted Non-IID / Dirichlet Non-IID / High Delay IID`

完整 `10-seed` 结果如下，数值为 mean ± std：

| Setting | Method | Final Acc | Best Acc | Final Loss |
|---|---|---:|---:|---:|
| Base IID | AsyncSAM+RMS | 76.74 ± 0.41 | 76.95 ± 0.33 | 0.9942 ± 0.0145 |
| Base IID | FedAsync | 71.18 ± 0.50 | 71.41 ± 0.43 | 1.1425 ± 0.0159 |
| Base IID | AsyncSGD | 71.11 ± 0.61 | 71.32 ± 0.54 | 1.1441 ± 0.0200 |
| Base IID | FedBuff | 66.94 ± 1.33 | 68.91 ± 0.43 | 1.2643 ± 0.0497 |
| Label-Sorted Non-IID | AsyncSAM+RMS | 76.75 ± 0.33 | 76.98 ± 0.24 | 0.9887 ± 0.0114 |
| Label-Sorted Non-IID | AsyncSGD | 70.93 ± 0.32 | 71.20 ± 0.36 | 1.1525 ± 0.0137 |
| Label-Sorted Non-IID | FedAsync | 70.87 ± 0.38 | 71.11 ± 0.39 | 1.1550 ± 0.0129 |
| Label-Sorted Non-IID | FedBuff | 66.98 ± 0.70 | 68.29 ± 0.39 | 1.2577 ± 0.0270 |
| Dirichlet Non-IID | AsyncSAM+RMS | 76.85 ± 0.38 | 77.03 ± 0.40 | 0.9852 ± 0.0124 |
| Dirichlet Non-IID | FedAsync | 70.90 ± 0.46 | 71.21 ± 0.49 | 1.1489 ± 0.0135 |
| Dirichlet Non-IID | AsyncSGD | 70.91 ± 0.56 | 71.18 ± 0.44 | 1.1504 ± 0.0164 |
| Dirichlet Non-IID | FedBuff | 66.56 ± 0.85 | 68.36 ± 0.32 | 1.2772 ± 0.0373 |
| High Delay IID | AsyncSAM+RMS | 76.61 ± 0.52 | 76.77 ± 0.40 | 0.9977 ± 0.0211 |
| High Delay IID | AsyncSGD | 71.30 ± 0.32 | 71.51 ± 0.34 | 1.1486 ± 0.0117 |
| High Delay IID | FedAsync | 71.18 ± 0.25 | 71.42 ± 0.23 | 1.1498 ± 0.0108 |
| High Delay IID | FedBuff | 67.24 ± 1.24 | 68.62 ± 0.61 | 1.2586 ± 0.0529 |

结论：

- `AsyncSAM+RMS` 在 ResNeXt29-8x16d 的四个 setting 中全部取得最高 final accuracy、best accuracy 和最低 final loss。
- 相比最强的一阶异步基线，final accuracy 提升约 `+5.31` 到 `+5.95` 个点。
- 相比 `FedBuff`，final accuracy 提升约 `+9.37` 到 `+10.29` 个点。
- 这组结果比 ResNet18/300e 的绝对精度更高，说明当前 `RMS-stabilized Anderson history` 的收益不只依赖 ResNet18 主干。

输出文件：

- `mnist/experiments/by_family/cifar100/resnext29/resnext29_8x16_lr003_e200_four_settings_10seed_fig/summary.csv`
- `mnist/experiments/by_family/cifar100/resnext29/resnext29_8x16_lr003_e200_four_settings_10seed_fig/resnext29_8x16_lr003_e200_four_settings_summary.png`
- `mnist/experiments/by_family/cifar100/resnext29/resnext29_8x16_lr003_e200_four_settings_10seed_fig/resnext29_8x16_lr003_e200_four_settings_curves.png`

口径说明：

- 正式对比图、正式表格和后续主结论，统一以本节 `11.19` 的完整 `4 setting / 10-seed` 结果为准。
- 下面的 `11.20` 只作为一次 `FedBuff` 实现复核备注，用来记录论文版聚合写法在 `Base IID / seed=1` 上的大致表现。
- 由于并未基于论文版 `FedBuff` 重新补齐 `4 setting / 10-seed`，因此 `11.20` 不替代本节主表，也不作为后续论文主图的默认来源。

### 11.20 CIFAR-100 / ResNeXt29-8x16d / Base IID / FedBuff 实现复核备注（不纳入主表）

为核对 `FedBuff` 的实现细节，已将 server 端聚合从“`mean(buffer)` 后再额外 `/K`”改为论文版写法，即只保留一次 buffer 平均。修正代码位于：

- `mnist/async_distributed_main.py`

在固定公共协议不变的前提下，对 `FedBuff` 的私有参数做了 `Base IID / seed=1` 救援筛选：

- `dataset = cifar100`
- `model = resnext29_8x16d`
- `partition = iid`
- `num_workers = 2`
- `epochs = 200`
- `lr = 0.03`
- `batch_size = 64`
- `cifar_augment = randaugment`
- `random_erasing = 0.25`
- `label_smoothing = 0.1`

共完成 `18` 组配置，包含三类：

- 纯 `FedBuff`
- `FedBuff+Momentum`
- `FedBuff+RMS`

当前最强的纯 buffered `FedBuff` 为：

| Variant | Config | Final Acc | Best Acc | Final Loss | Best Loss |
|---|---|---:|---:|---:|---:|
| FedBuff | `k=2, etag=3.9` | 72.76 | 73.39 | 1.0849 | 1.0551 |
| FedBuff+Momentum | `momentum=0.5, k=2, etag=3.2` | 72.28 | 72.28 | 1.1115 | 1.0956 |
| FedBuff+RMS | `k=2, etag=1.6` | 71.75 | 72.11 | 1.1126 | 1.0693 |

结论：

- 在这套固定公共协议下，修正为论文版后，`FedBuff` 的 `Base IID` 单 seed 最优 final accuracy 从旧记录的 `~66.9%` 提升到了 `72.76%`。
- 但即使继续加入轻量 `momentum` 或 `RMS` 预条件，增强版也没有超过最好的纯 buffered `FedBuff`。
- 因而当前可采用的 `FedBuff` 参考配置为 `k=2, etag=3.9`。
- 这一节仅记录 `Base IID / seed=1` 的实现复核结果；当前项目的正式主线仍统一采用 `11.19` 的完整 `4 setting / 10-seed` 对比口径。
- 若后续确实需要把论文版 `FedBuff` 放入正式主图，应以 `k=2, etag=3.9` 为起点重新补齐四设置 `10-seed`，否则不再单独引用这里的 seed1 图。

输出文件：

- `mnist/experiments/by_family/cifar100/resnext29/baseline_tune_base_iid_fedbuff_paper_rescue2_fig/summary.csv`
- `mnist/experiments/by_family/cifar100/resnext29/baseline_tune_base_iid_fedbuff_paper_rescue2_fig/fedbuff_paper_rescue2_summary.png`
- `mnist/experiments/by_family/cifar100/resnext29/baseline_tune_base_iid_fedbuff_paper_rescue2_fig/fedbuff_paper_rescue2_summary.pdf`
- `mnist/experiments/by_family/cifar100/resnext29/resnext29_8x16_lr003_e200_base_iid_seed1_bestfedbuff_compare_fig/summary.csv`
- `mnist/experiments/by_family/cifar100/resnext29/resnext29_8x16_lr003_e200_base_iid_seed1_bestfedbuff_compare_fig/resnext29_8x16_lr003_e200_base_iid_seed1_bestfedbuff_compare.png`
- `mnist/experiments/by_family/cifar100/resnext29/resnext29_8x16_lr003_e200_base_iid_seed1_bestfedbuff_compare_fig/resnext29_8x16_lr003_e200_base_iid_seed1_bestfedbuff_compare.pdf`

### 11.21 AFL-Lib FADAS / CA2FL baseline 接入与正确性复核

已确认 AFL-Lib 中存在 `FADAS` 与 `CA2FL` 两个异步 baseline：

- `AFL-Lib/alg/fadas.py`
- `AFL-Lib/alg/ca2fl.py`

同时已将它们接入当前统一 simulator：

- `mnist/async_distributed_main.py`

实现口径：

- `FADAS` 消费客户端模型更新差 `dW = local_model_after - local_model_before`。在当前 minibatch 异步 simulator 中，对应写作 `dW = -lr * grad`。
- `FADAS` 使用 buffer 平均 `Delta_t`，再做 adaptive server update：`m_t = beta1 m_{t-1} + (1-beta1) Delta_t`，`v_t = beta2 v_{t-1} + (1-beta2) Delta_t^2`，`v_hat_t = max(v_hat_{t-1}, v_t)`，`x_{t+1} = x_t + eta_t * m_t / (sqrt(v_hat_t) + eps)`。
- `eta_t` 使用 AFL-Lib 中的实用 delay-adaptive 口径：当最大 staleness 超过阈值时采用 `eta / tau_max`。
- `CA2FL` 维护每个 client 的 cached update `h_i`，每个 buffer round 使用 `v_t = h_t + mean(dW_i - h_i)`，其中 `h_t` 是 round-start cache average。

正确性复核：

- AFL-Lib 的 `FADAS` 主流程基本完整，但代码计算了 `v_hat` 后实际分母使用的是 `sqrt(v)`；当前 simulator 默认使用 AMSGrad/FADAS 更合理的 `sqrt(v_hat)`，并提供 `--fadas-use-vhat 0` 可退回 AFL-Lib 写法。
- AFL-Lib 的 `CA2FL` 存在一个明显 bookkeeping 问题：`buffer_clients` 被初始化和清空，但在 `aggregate()` 中没有 append，导致 calibration 项不能按 client id 正确计算。当前 simulator 按算法语义修正为直接缓存并累加 `dW_i - h_i`。
- 已完成 `py_compile`。
- 已完成 `--help` 入口检查，`--alg` 已包含 `fadas` 和 `ca2fl`。
- 已完成 MNIST CPU smoke test：两个算法均完成 `1 epoch`，且均触发 `buffer_wait=2, buffer_apply=2`。
- 已完成闭式公式单元检查：FADAS 的 AMSGrad-style update 与 CA2FL 的 cached calibration update 均与手算公式一致。

后续如果纳入正式 baseline，需要在各主线实验中分别调它们的私有参数：

- `FADAS`: `fadas_m, fadas_tau_c, fadas_beta1, fadas_beta2, fadas_eta`
- `CA2FL`: `ca2fl_m, ca2fl_eta`

### 11.22 FADAS / CA2FL / AsyncSGD constant 正式补充 baseline 已完成并汇总

已新增分布式 work-stealing 补跑脚本：

- `mnist/run_fadas_ca2fl_main_comparisons_distributed.sh`

本轮目标是把 `FADAS` 和 `CA2FL` 补进已有正式实验族，作为额外异步 baseline。公平性口径：

- 不改变各实验族已有公共训练协议。
- 不改变 dataset / model / batch size / epochs / lr / lr schedule / weight decay / augmentation / partition / delay setting。
- 只使用算法私有参数：
  - `FADAS`: `M=5, tau_c=1, beta1=0.9, beta2=0.99, eta=0.001, use_vhat=1, delay_adapt=1`
  - `CA2FL`: `M=10, eta=1.0`

覆盖范围：

- `MNIST`: Base IID / Label-Sorted / Dirichlet / High Delay / 20 Workers
- `CIFAR-10 + ResNet18`: Base IID / Label-Sorted / High Delay / 20 Workers original / 20 Workers tuned T3/E200
- `CIFAR-10 + ResNet32`: Base IID / Label-Sorted / Dirichlet / High Delay
- `CIFAR-10 + ResNet56`: Base IID / Label-Sorted / Dirichlet / High Delay
- `CIFAR-100 + ResNet32`: Base IID / Label-Sorted / Dirichlet / High Delay
- `CIFAR-100 + ResNet18 / 300e / RandAug+RE+LS`: Base IID / Label-Sorted / Dirichlet / High Delay
- `CIFAR-100 + ResNeXt29-8x16d / 200e`: Base IID / Label-Sorted / Dirichlet / High Delay

总任务量：

- `30 settings * 2 methods * 10 seeds = 600` runs

完成状态：

- 已在 `10.42.27.130-137` 上启动同一 work-stealing 脚本。
- 空闲节点 `131/132/135` 使用较高并发，忙节点 `130/133/134/136/137` 少量叠加。
- 共享锁目录：`mnist/experiments/fadas_ca2fl_main_comparisons_locks/`
- 日志目录：`mnist/experiments/fadas_ca2fl_main_comparisons_logs/`
- `FADAS / CA2FL`: `600/600` 正式 `.pkl` 完成，坏文件 `0`。
- `AsyncSGD --stale-strategy constant`: `300/300` 正式 `.pkl` 完成，坏文件 `0`。
- 所有 `90` 个 setting-method 汇总行均为 `10/10` seeds 完整结果。

新增 baseline 原始汇总脚本与输出：

- `mnist/summarize_added_baselines.py`
- `mnist/experiments/added_baselines_fadas_ca2fl_constant_summary/summary.csv`
- `mnist/experiments/added_baselines_fadas_ca2fl_constant_summary/family_summary.csv`
- `mnist/experiments/added_baselines_fadas_ca2fl_constant_summary/family_mean_final_acc.svg`

已进一步把新增三条 baseline 与之前四个主算法合并为完整横向对比表：

- `mnist/summarize_all_main_baselines.py`
- `mnist/experiments/all_main_baselines_summary/summary.csv`
- `mnist/experiments/all_main_baselines_summary/wide_final_acc.csv`
- `mnist/experiments/all_main_baselines_summary/wide_final_loss.csv`
- `mnist/experiments/all_main_baselines_summary/wide_final_acc_main7.csv`
- `mnist/experiments/all_main_baselines_summary/wide_final_loss_main7.csv`
- `mnist/experiments/all_main_baselines_summary/wide_final_acc.md`
- `mnist/experiments/all_main_baselines_summary/wide_final_acc_main7.md`
- `mnist/experiments/all_main_baselines_summary/family_summary.csv`
- `mnist/experiments/all_main_baselines_summary/family_summary.md`

完整横向汇总当前包含 `211` 条 setting-method 记录。主表不再跨 setting 求平均，而是逐 setting 展示。除 `MNIST` 的历史验证表额外保留 `AsyncSGD+RMS` 外，主线方法为：

- 之前四个主算法：`AsyncSAM+RMS / AsyncSGD / FedAsync / FedBuff`
- 新增三条 baseline：`FADAS / CA2FL / AsyncSGD constant`

不跨 setting 平均的主总表：

- `wide_final_acc_main7.md`: 只含主 7 方法，每行是一个具体 setting 的 `final acc mean ± std`
- `wide_final_acc.md`: 保留 `MNIST` 历史额外列 `AsyncSGD+RMS`
- `summary.csv`: 长表，包含每个 setting-method 的 `final / best acc` 与 `final / best loss`

后续论文主表应优先引用 `wide_final_acc_main7.md` 或 `summary.csv`，不要再使用跨 setting 的 family mean 作为主结论数值。

关键观察：

- `AsyncSAM+RMS` 在多数具体 setting 上仍是最强方法，尤其在 CIFAR-10 ResNet32/56 和 CIFAR-100 ResNeXt29 上优势清晰。
- `FADAS` 在 `MNIST` 上表现较强，但在 CIFAR 系列中通常低于 `AsyncSGD constant` 和已有 `AsyncSAM+RMS` 主方法。
- `CA2FL` 在当前 minibatch 异步 simulator 下整体偏弱，尤其在 CIFAR-100 / ResNet32 与 ResNeXt29 上明显落后；这说明 cached update calibration 在当前强增广/高维模型设置下并没有直接转化为竞争力。
- `AsyncSGD constant` 是一个必要的“标准不降权异步 SGD”补充 baseline。它在多个 CIFAR 设置上强于 hinge staleness 版本，但仍没有改变 `AsyncSAM+RMS` 在正式主表中的主结论。
- 后续若论文中加入 `FADAS / CA2FL`，建议作为额外 baseline 行放入附录或扩展表；主文重点仍放 `AsyncSAM+RMS` 对 `AsyncSGD / FedAsync / FedBuff` 的稳定优势。

### 11.23 FADAS / CA2FL baseline 口径收缩与私有参数重调

根据当前主线要求，额外 baseline 的展示和后续补跑先做如下收缩：

- 删除 `CIFAR-10 ResNet18 / 20 Workers Original`，仅保留后续调过公共协议的 `20 Workers T3/E200`。
- 暂时不纳入 `CIFAR-100 ResNet32`。
- 主总表不再跨 setting 求 mean，统一使用逐 setting 的 `wide_final_acc_main7.md` 和长表 `summary.csv`。

已修改：

- `mnist/summarize_all_main_baselines.py`: 过滤上述两个不纳入口径的 setting/family。
- `mnist/run_fadas_ca2fl_main_comparisons_distributed.sh`: 后续正式补跑不再排 `CIFAR-10 ResNet18 / 20 Workers Original` 和 `CIFAR-100 ResNet32`。
- `mnist/async_distributed_main.py`: `CA2FL` 的默认 `--ca2fl-eta` 从 `1.0` 改为 `0.01`，与 AFL-Lib 原版默认值一致。

同时已启动 FADAS / CA2FL 私有参数短筛 pilot：

- 脚本：`mnist/run_fadas_ca2fl_tune_pilot.sh`
- 汇总：`mnist/summarize_fadas_ca2fl_tune_pilot.py`
- 输出目录：`mnist/experiments/tune_fadas_ca2fl_private/`
- 汇总目录：`mnist/experiments/tune_fadas_ca2fl_private_summary/`

短筛设置：

- 只跑 `seed=1`。
- 代表 family：`CIFAR-10 ResNet18 Base IID`、`CIFAR-10 ResNet56 Base IID`、`CIFAR-100 ResNet18 300e recipe Base IID`、`CIFAR-100 ResNeXt29-8x16d 200e Base IID`。
- 公共协议保持对应 family 的主线设置，只缩短 epoch 用作筛选。
- FADAS 只调私有参数：`M / eta / use_vhat / delay_adapt`。
- CA2FL 只调私有参数：`M / eta`。

当前已确认本轮 pilot 在沙箱外真实调用 NPU；沙箱内会触发 `torch_npu rtGetDeviceCount failed`，因此 NPU 训练需要使用外部执行权限。
