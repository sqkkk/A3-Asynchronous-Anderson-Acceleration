# AsyncSAM+RMS 算法说明与合理性分析

本文档说明当前实验中使用的 `AsyncSAM+RMS` 算法。这里的 `SAM` 指的是 **Stochastic Anderson Mixing**，不是 Sharpness-Aware Minimization。当前实现位于：

- `mnist/async_distributed_main.py`
- 核心类：`AsyncDistributedSAMServer`
- 关键函数：`_compute_preconditioned_residual(...)` 和 `update(...)`

## 1. 一句话概括

`AsyncSAM+RMS` 是一个中心服务器侧的异步优化器。每次服务器收到一个 worker 返回的梯度或梯度类更新后，先构造一个带 staleness 权重的基础异步 SGD 步，再用 RMS 二阶矩预条件器做坐标级缩放，然后用 Stochastic Anderson Mixing 从最近的服务器更新历史中估计一个加速步。最后为了稳定性，将加速步和基础步做保守混合。

更直观地说：

- `AsyncSGD`：收到一个梯度，就沿这个梯度走一步。
- `AsyncSAM`：不只看当前这一步，还利用最近几次“参数变化”和“残差变化”的关系，做一个近似二阶的 Anderson 加速。
- `AsyncSAM+RMS`：在 `AsyncSAM` 前面加上 RMS 预条件，使每个参数维度的步长按历史梯度尺度自适应缩放，从而在异步和非 IID 噪声下更稳定。

## 2. 异步训练框架

服务器维护全局参数向量：

```text
x_k
```

第 `i` 个 worker 在某个时间拿到服务器快照：

```text
x_s
```

随后 worker 在本地 batch 上计算梯度类信息：

```text
g_i(x_s)
```

由于是异步系统，worker 返回时服务器可能已经更新了多次。代码中定义 staleness：

```text
tau = current_server_step - dispatch_step
```

服务器收到这个 stale 梯度后，不等待其它 worker，而是立即尝试更新当前全局模型 `x_k`。

## 3. 基础异步 SGD 步

首先根据 staleness 计算权重：

```text
w(tau) = staleness_weight(tau)
```

当前默认可以是：

```text
constant: w(tau) = 1
poly:     w(tau) = 1 / (tau + 1)^a
hinge:    tau <= b 时为 1，否则为 1 / (a * (tau + b) + 1)
```

然后用当前 server 学习率 `eta_k` 构造基础残差，也就是基础异步下降步：

```text
b_k = - eta_k * w(tau) * g_k
```

如果没有 RMS、没有 Anderson mixing，那么基础更新就是：

```text
y_k = x_k + b_k
```

这里 `b_k` 是一个 server-side update residual，也可以理解成“当前收到的异步梯度所诱导的一步基础下降方向”。

## 4. RMS 预条件

`AsyncSAM+RMS` 不直接使用 `b_k`，而是先维护梯度平方的指数滑动平均：

```text
v_k = beta_rms * v_{k-1} + (1 - beta_rms) * g_k^2
```

然后构造逐坐标分母：

```text
d_k = sqrt(v_k) + eps
```

如果设置了最小分母，则执行：

```text
d_k = max(d_k, min_denom)
```

最终的 RMS 预条件残差为：

```text
r_k = - eta_k * w(tau) * g_k / d_k
```

因此基础步从 `x_k + b_k` 变成：

```text
y_k = x_k + r_k
```

在代码中，`r_k` 对应 `_compute_preconditioned_residual(...)` 的返回值。

### RMS 的作用

RMS 的核心作用是把“梯度幅值特别大或波动特别大的坐标”自动缩小，把“梯度幅值较小的坐标”相对放大。它和 RMSProp/Adam 中的二阶矩思想相近，但这里它被放在异步服务器聚合侧，用来稳定来自不同 worker、不同 staleness、不同数据分布的梯度类更新。

这在异步联邦或异步分布式里很重要，因为不同 worker 返回的梯度可能具有：

- 不同的数据分布，尤其 non-IID。
- 不同的延迟，导致梯度对应旧模型。
- 不同的随机 batch 噪声。
- 不同坐标尺度和曲率。

RMS 预条件可以降低这些因素带来的坐标尺度不均衡，让后续 Anderson mixing 看到的 residual 更稳定。

## 5. residual momentum

如果设置了 `sam_momentum > 0`，算法还会对预条件残差做动量平滑：

```text
m_k = mu * m_{k-1} + (1 - mu) * r_k
```

并做 bias correction：

```text
r_k = m_k / (1 - mu^t)
```

这一步不是 Adam 的完整一阶动量更新，而是对 server residual 做平滑。它的目的主要是降低异步随机梯度的短期震荡，让 Anderson 历史矩阵更干净。

在不少 CIFAR 主线实验里，常见配置包括：

```text
sam_momentum = 0.9
sam_precond = rms
sam_precond_beta = 0.90
sam_precond_init = 1.0
sam_precond_min_denom = 0.1 / 0.2 / 0.3   (随 recipe 变化)
```

## 6. Stochastic Anderson Mixing 历史

基础步为：

```text
y_k = x_k + r_k
```

如果只使用 RMS 预条件异步 SGD，则可以直接令：

```text
x_{k+1} = y_k
```

但 `AsyncSAM+RMS` 还会尝试利用历史信息做 Anderson mixing。

当前实现维护两类历史差分：

```text
Delta x_k = x_k - x_{k-1}
Delta r_k = r_k - r_{k-1}
```

为了降低随机噪声，代码里不是直接存原始差分，而是使用指数平滑后的差分：

```text
dX_k = gamma * dX_{k-1} + (1 - gamma) * (x_k - x_{k-1})
dR_k = gamma * dR_{k-1} + (1 - gamma) * (r_k - r_{k-1})
```

然后将最近若干个 `dX_k` 和 `dR_k` 存成矩阵：

```text
X = [dX_{k-m+1}, ..., dX_k]
R = [dR_{k-m+1}, ..., dR_k]
```

其中 `m` 由 `sam_hist_length` 控制。当前主线常用：

```text
sam_hist_length = 4
```

## 7. Anderson mixing 的求解形式

当前实现采用 Stochastic Anderson Mixing 中的正则化最小二乘形式。令当前 residual 为 `r_k`，求一个系数向量 `c_k`：

```text
(R^T R + delta * X^T X + ridge * I) c_k = R^T r_k
```

其中：

```text
delta = sam_damp * ||r_k||^2 / (||Delta x_ref||^2 + eps)
```

直观理解：

- `R^T R` 利用 residual 变化拟合当前 residual。
- `X^T X` 是 damping 项，避免历史参数差分导致过大的外推。
- `ridge * I` 是额外数值正则项，避免矩阵病态。
- `c_k` 是 Anderson mixing 的历史校正系数。

求得 `c_k` 后，构造加速方向：

```text
d_aa = beta * r_k - (alpha * X + alpha * beta * R) c_k
```

然后候选加速模型为：

```text
x_aa = x_k + d_aa
```

这一步可以理解为：基础 residual `r_k` 给出一阶下降方向，历史矩阵 `(X, R)` 给出局部 secant 信息，用来修正当前步，使它更接近“考虑了局部曲率后的更新”。

## 8. 接受、拒绝和保守混合

异步环境中，Anderson mixing 很容易因为 stale 梯度、non-IID 噪声或病态历史矩阵产生过大外推。因此当前实现加入了多层 safeguard。

### 8.1 历史过滤

如果设置了：

```text
sam_max_history_staleness = 2
```

那么过期太久的历史不会进入 Anderson 历史矩阵。这样可以减少“很旧模型上的梯度变化”污染当前局部几何估计。

需要注意的是，当前到达的更新如果 `tau` 过大，也不会被写入新的历史槽位；但这并不意味着服务器一定直接回退到基础步。若已有足够的新鲜历史，代码仍可能构造一个 Anderson 候选，只是随后会通过更保守的 base mixing 来降低外推风险。

### 8.2 周期性启用

如果设置：

```text
sam_period = 2
```

则不是每次 fresh history arrival 都做 Anderson mixing，而是按内部 `aa_step` 周期性触发。这能降低计算成本，也能给历史 residual 留出足够变化。

### 8.3 病态矩阵拒绝

如果线性系统条件数太大：

```text
cond(system) > sam_max_cond
```

则拒绝加速，回退到基础步：

```text
x_{k+1} = y_k
```

当前常用：

```text
sam_max_cond = 10000
```

### 8.4 步长比例限制

如果加速步过大，会先被裁剪；裁剪后仍然过大时，再拒绝。核心约束是：

```text
||d_aa|| <= sam_max_step_ratio * ||r_k||
```

当前常用：

```text
sam_max_step_ratio = 1.0
```

这意味着 Anderson 加速方向的范数不能明显大于当前基础 residual，避免激进外推。

### 8.5 anchor 限制

如果设置：

```text
sam_anchor_tol = 1.0
```

则约束的是 Anderson 候选步长本身，而不是 `x_aa` 到 `y_k` 的欧氏距离。更准确地说，代码检查的是：

```text
||d_aa|| <= sam_anchor_tol * ||r_k||
```

也就是要求 Anderson 校正步的范数不要相对基础步过大。

### 8.6 与基础步混合

即使 `x_aa` 被接受，当前主线仍然会和基础步混合：

```text
x_{k+1} = (1 - rho) * x_aa + rho * y_k
```

其中基础混合比例的下界由：

```text
rho = sam_base_mix
```

在较 stale 的更新上，代码还可能进一步增大混合比例：

```text
rho = max(rho, sam_stale_base_mix)
rho = max(rho, tau_dependent_mix)
```

因此 `sam_base_mix` 更像“最小保守混合比例”，而不是唯一的混合来源。

不少 CIFAR 主线实验中常用：

```text
sam_base_mix = 0.8
```

这意味着最终更新中 `80%` 来自基础 RMS 异步步，`20%` 来自 Anderson 校正。这个设计非常保守，但在异步 non-IID 场景下很关键，因为它让算法不会完全相信历史外推。

## 9. 完整算法伪代码

```text
输入:
  当前服务器参数 x_k
  worker 返回梯度 g_k
  staleness tau
  历史矩阵 X, R

1. 计算 staleness 权重
   w = staleness_weight(tau)

2. 构造基础异步残差
   b_k = - eta_k * w * g_k

3. RMS 预条件
   v_k = beta_rms * v_{k-1} + (1 - beta_rms) * g_k^2
   denom = sqrt(v_k) + eps
   denom = max(denom, min_denom)
   r_k = - eta_k * w * g_k / denom

4. residual momentum
   m_k = mu * m_{k-1} + (1 - mu) * r_k
   r_k = m_k / (1 - mu^t)

5. 基础候选
   y_k = x_k + r_k

6. 更新 SAM 历史
   dX_k = EMA(x_k - x_{k-1})
   dR_k = EMA(r_k - r_{k-1})
   将 dX_k, dR_k 写入历史矩阵 X, R

7. 如果历史不足、周期未到，或有效历史不足:
   返回 y_k
   如果当前更新太 stale:
     不把当前更新写入历史；若已有足够新鲜历史，后续仍可继续尝试 Anderson 候选

8. 求 Anderson mixing 系数
   (R^T R + delta X^T X + ridge I) c_k = R^T r_k

9. 构造加速方向
   d_aa = beta * r_k - (alpha X + alpha beta R) c_k
   x_aa = x_k + d_aa

10. safeguard
   若非有限、条件数太大、步长过大、与基础步方向不一致:
     返回 y_k

11. 保守混合
   x_{k+1} = (1 - rho) * x_aa + rho * y_k
```

## 10. 为什么这个算法在异步分布式场景中合理

### 10.1 异步系统只能利用到达的梯度

在同步分布式训练中，服务器通常等待一组 worker 后求平均梯度。但异步系统中，服务器每次只收到当前完成 worker 的更新。如果强行等待所有 worker，就失去了异步训练的意义。

`AsyncSAM+RMS` 的设计是 event-driven 的：收到一个 worker 的梯度就更新一次。它不要求连续时间段的完整梯度，也不要求所有 worker 同步，只利用当前到达的梯度和最近的服务器更新历史。

### 10.2 RMS 处理异步梯度的尺度噪声

异步 non-IID 下，某个 worker 的梯度可能在某些坐标特别大，这会导致 server 更新震荡。RMS 预条件通过 `sqrt(v_k)` 缩放梯度，使不同坐标的更新尺度更接近。

这相当于在 server 侧加入一个轻量的自适应度量：

```text
g_k -> g_k / sqrt(E[g^2])
```

因此，它不是简单调小全局学习率，而是逐坐标地调节步长。

### 10.3 SAM 利用最近历史近似局部曲率

Anderson mixing 的核心思想是：如果最近几步中，参数变化 `Delta x` 和 residual 变化 `Delta r` 呈现稳定关系，那么可以利用这种关系估计一个更好的下一步。

在深度网络中不可能显式求 Hessian，也不适合在异步服务器上做昂贵二阶优化。但 `X` 和 `R` 的历史矩阵提供了低秩 secant 信息，相当于用最近几步的行为估计局部几何。

因此 `AsyncSAM+RMS` 可以看作：

```text
异步 SGD 基础步
+ RMS 坐标预条件
+ 低秩历史 secant 校正
+ 异步稳定性 safeguard
```

### 10.4 保守混合使算法不脱离基础优化方向

当前实验中 `sam_base_mix = 0.8`，说明最终步大部分仍来自基础 RMS 异步 SGD：

```text
80% base step + 20% Anderson correction
```

这让方法具备两个性质：

- 如果 Anderson 历史可靠，它可以提供额外加速。
- 如果 Anderson 历史不可靠，基础步仍然占主导，训练不会轻易崩。

这一点尤其适合异步场景，因为 staleness 和 non-IID 会让历史矩阵比同步单机实验更噪。

### 10.5 stale 历史过滤降低错误 secant 信息

如果某个历史 residual 来自很旧的模型快照，它和当前 `x_k` 周围的局部几何可能已经不一致。`sam_max_history_staleness` 过滤机制使 Anderson 历史更接近当前局部区域；而对当前过 stale 的到达，代码也会避免把它写进历史，只允许它在已有新鲜历史基础上参与更保守的混合更新。

这不是改变 Anderson mixing 的核心思想，而是在异步系统中保证历史信息“还属于同一个局部问题”。

## 11. 和其它算法的关系

### 11.1 和 AsyncSGD 的关系

`AsyncSGD` 使用：

```text
x_{k+1} = x_k - eta_k * w(tau) * g_k
```

`AsyncSAM+RMS` 则使用：

```text
r_k = - eta_k * w(tau) * g_k / sqrt(v_k)
x_{k+1} = AndersonMix(x_k, r_k, history)
```

因此它是在 AsyncSGD 的基础上加入 RMS 预条件和 Anderson 历史校正。

### 11.2 和 Adam/RMSProp 的关系

RMS 部分与 RMSProp/Adam 的二阶矩估计相似：

```text
v_k = beta v_{k-1} + (1 - beta) g_k^2
```

但当前算法不是完整 Adam：

- 没有使用 Adam 的参数级一阶矩更新公式。
- 一阶动量是对 residual 做平滑，而不是标准 Adam 的 `m/sqrt(v)` 形式。
- RMS 发生在服务器侧，输入是异步返回的 gradient-like update。
- RMS 后面还有 Anderson mixing。

所以更准确的名字是：

```text
Async Stochastic Anderson Mixing with RMS preconditioning
```

### 11.3 和 AsyncAA 的关系

当前代码里也有 `asyncaa`。它使用的是更标准的 GD-style Anderson acceleration：

```text
y_i = x_i + r_i
x_next = sum_i alpha_i y_i
```

而 `AsyncSAM+RMS` 使用的是 Stochastic Anderson Mixing 的差分形式：

```text
X = [Delta x_i]
R = [Delta r_i]
d_aa = beta r_k - (alpha X + alpha beta R)c_k
```

两者都利用 residual history，但输出形式不同。`AsyncSAM+RMS` 当前实验效果更好，主要是因为它同时结合了 RMS 预条件、residual momentum、保守 base mix 和异步 stale safeguard。

### 11.4 和 FedAC 的关系

FedAC 更像是带控制变量和自适应聚合的联邦算法，当前实现还涉及 client-side local training、control variate 和二阶矩式 server update。

`AsyncSAM+RMS` 不依赖 FedAC 的控制变量机制，也不要求 worker 做多步本地训练。它直接消费异步返回的 gradient-like update，因此更贴近当前异步 mini-batch server update scaffold。

## 12. 当前实验中常用参数

不同实验会略有差异，而且代码默认值并不是下面这组数；下面更准确地说，是“当前正式实验里经常使用的 recipe 取值范围”：

| 参数 | 含义 | 常用值 |
|---|---|---:|
| `sam_precond` | 是否启用预条件 | `rms` |
| `sam_precond_beta` | RMS 二阶矩 EMA 系数 | `0.90` |
| `sam_precond_init` | RMS 二阶矩初值 | `1.0` |
| `sam_precond_min_denom` | RMS 分母下界 | `0.1 / 0.2 / 0.3` |
| `sam_momentum` | residual momentum | `0.9` |
| `sam_hist_length` | Anderson 历史长度 | `4` |
| `sam_period` | Anderson 触发周期 | `2` |
| `sam_base_mix` | 基础步混合比例下界 | `0.5 - 0.8`，主线多为 `0.8` |
| `sam_max_history_staleness` | 允许进入历史的最大 staleness | `2` |
| `sam_stale_base_mix` | 过 stale 更新时的最小 base mixing | `0.2`（高延迟/高并发时常用） |
| `sam_max_cond` | 条件数上限 | `10000`（部分 rescue sweep 更大） |
| `sam_max_step_ratio` | 加速步相对基础步的最大比例 | `1.0` |
| `sam_anchor_tol` | Anderson 步长相对基础步的范数约束 | `1.0` |

其中最重要的几个参数是：

- `sam_precond_beta`: 控制 RMS 记忆长度，越大越平滑，越小越敏感。
- `sam_base_mix`: 控制 Anderson 校正的激进程度，越大越保守；在更 stale 的到达上，还可能叠加 `sam_stale_base_mix` 或 `sam_tau_base_mix`。
- `sam_hist_length`: 控制历史矩阵大小，太小信息不足，太大容易混入不一致历史。
- `sam_max_history_staleness`: 控制异步历史过滤，越小越稳但可用历史更少。

## 13. 局限性

`AsyncSAM+RMS` 的合理性来自“异步基础步 + RMS 稳定化 + 低秩历史加速”，但它仍然是近似方法。

主要近似包括：

- worker 返回的是 stochastic mini-batch gradient，不是全量梯度。
- 梯度是在旧快照 `x_s` 上计算的，不是在当前 `x_k` 上计算的。
- non-IID 下不同 worker 的 gradient direction 可能代表不同局部目标。
- Anderson 历史矩阵不一定来自同一个平滑固定点映射。
- RMS 的二阶矩估计混合了不同 worker、不同 staleness、不同数据分布的梯度。

因此 safeguards 不是工程补丁，而是算法在异步分布式场景中可用的必要条件。

## 14. 适合写进论文的方法表述

可以将 `AsyncSAM+RMS` 表述为：

```text
We propose an asynchronous server-side stochastic Anderson mixing method with
RMS preconditioning. Upon each stale worker update, the server forms a
staleness-weighted base descent residual, normalizes it by an exponential
moving average of squared gradients, and applies a safeguarded Anderson-type
secant correction using recent residual and iterate differences. The final
update is conservatively blended with the base RMS-preconditioned step to
retain stability under asynchronous and non-IID updates.
```

中文表述可以写为：

```text
本文采用一种服务器侧异步随机 Anderson mixing 方法。服务器每收到一个
stale worker 更新，即构造带 staleness 权重的基础下降 residual，并用梯度
平方指数滑动平均进行 RMS 预条件。随后，服务器利用最近参数差分和 residual
差分形成低秩 secant 历史，求解带阻尼和正则化的 Anderson mixing 系数，
生成候选加速步。为适应异步和 non-IID 场景，算法加入历史 staleness 过滤、
条件数检查、步长比例约束，并将候选加速步与基础 RMS 步保守混合。
```

## 15. 小结

`AsyncSAM+RMS` 的核心不是简单“把历史方向平均一下”，而是：

```text
1. 当前异步梯度给出基础下降 residual。
2. RMS 预条件降低坐标尺度和异步噪声。
3. Anderson/SAM 历史差分提供低秩 secant 校正。
4. 多个 safeguard 防止异步 stale 历史导致不稳定。
5. base mix 让最终更新始终锚定在基础 RMS 下降步附近。
```

因此，它既保留了异步训练“收到就更新”的效率，也引入了比 AsyncSGD 更强的历史几何利用能力。
