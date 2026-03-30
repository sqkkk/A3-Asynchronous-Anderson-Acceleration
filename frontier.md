# Phase 1: Frontier - AsyncAA Research

## 调研日期
2026-03-29

## 研究方向
基于异步历史更新池的 Anderson 加速分布式优化

## 搜索关键词
1. Anderson Acceleration fixed-point iteration optimization
2. asynchronous federated learning FedAsync FedBuff
3. federated learning client drift SCAFFOLD FedProx heterogeneous

## 发现的关键论文

### Anderson Acceleration 基础与进展

| 论文 | 作者/来源 | 关键贡献 | 状态 |
|-------|-----------|---------|------|
| Improved Convergence Factor of Windowed Anderson Acceleration | arXiv:2311.02490 | 窗口化 AA 的收敛因子分析 | 待读 |
| A Proof That Anderson Acceleration Improves the Convergence Rate | SIAM J. Numer. Anal. | AA 改善收敛速率的理论证明 | 重要基础 |
| Acceleration methods for fixed-point iterations | Acta Numerica | AA 方法综述 | 综述性论文 |
| Convergence Analysis of the Alternating Anderson-Picard Method | arXiv:2407.10472 | 交替式 AA 收敛分析 | 重要参考 |
| Non-stationary Anderson acceleration with optimized damping | TU Delft | 非定常 AA 与阻尼优化 | 相关工作 |

### 异步联邦学习最新进展

| 论文 | 作者/来源 | 关键贡献 | 状态 |
|-------|-----------|---------|------|
| FedBuff: Federated Learning with Buffered Asynchronous Aggregation | AISTATS 2022 | 缓冲异步聚合，结合同步与异步优点 | 核心基准 |
| FedAsync: Asynchronous Federated Optimization | arXiv 2019 | 纯异步联邦优化 | 核心基准 |
| FedFa: A Fully Asynchronous Training Paradigm | IJCAI 2024 | 完全异步训练范式 | 最新工作 |
| Efficient Asynchronous Federated Learning with... | AAAI 2024 | 高效异步联邦学习 | 待读 |
| FedASMU: Efficient Asynchronous Federated Learning | Fudan 2024 | 异步联邦学习方法 | 待读 |
| FedCompass: efficient cross-silo federated learning | ICLR 2024 | 跨孤岛联邦学习 | 待读 |

### 异构联邦学习与 Client Drift

| 论文 | 作者/来源 | 关键贡献 | 状态 |
|-------|-----------|---------|------|
| SCAFFOLD: Stochastic Controlled Averaging | ICML 2020 | Control variates 校正 client drift | 核心基准 |
| FedProx: Federated Optimization in Heterogeneous Networks | MLSys 2020 | Proximal 正则化处理异构性 | 核心基准 |
| FedNova: Tackling Objective Inconsistency | NeurIPS 2020 | 异构场景下的目标不一致性问题 | 重要参考 |
| Amplified SCAFFOLD | arXiv 2024 | 周期性客户端参与下的 SCAFFOLD 改进 | 最新工作 |
| FedBSS: Sample-level Client Drift Mitigation | AAAI 2024 | 样本级别 client drift 缓解 | 待读 |
| FedProto: Federated Prototype Learning | ResearchGate 2022 | 跨异构客户端的原型学习 | 相关工作 |

### AA + 联邦学习/分布式优化

| 论文 | 作者/来源 | 关键贡献 | 状态 |
|-------|-----------|---------|------|
| FedOSAA: Improving Federated Learning with One-Step Anderson Acceleration | arXiv 2025 | 联邦学习中的单步 AA | **最相关工作** |
| Anderson Acceleration for Distributed Constrained Optimization | arXiv 2026 | 时变网络上的分布式约束优化 AA | **最相关工作** |
| Accelerating Federated Edge Learning | Josh Nguyen Blog | 联邦边缘学习加速 | 相关工作 |
| Stochastic Anderson Mixing for Nonconvex Stochastic Optimization | NeurIPS 2021 | 非凸随机优化中的 AA | 重要基础 |

## 关键趋势与洞察

### 1. Anderson Acceleration 研究趋势
- **理论进展**: 窗口化 AA 的收敛因子分析有了更精确的结果
- **安全化**: Ridge 正则化、restart 机制、alternating trigger 是标配
- **应用拓展**: 从固定点迭代拓展到随机优化、分布式优化

### 2. 异步联邦学习趋势
- **FedBuff 成为主流**: 缓冲异步聚合结合了同步与异步的优点
- **完全异步**: FedFa 等工作探索无障碍异步更新
- **稳定性与效率平衡**: 如何在 wall-clock 时间上实现加速

### 3. 异构联邦学习趋势
- **Control variates 是主流**: SCAFFOLD 及其变体 (Amplified SCAFFOLD)
- **多层次校正**: 从样本级别到客户端级别的 drift 缓解
- **目标不一致性**: FedNova 等工作关注异构场景下的目标偏移

### 4. 与本研究最相关的工作

#### FedOSAA (2025)
- **核心思想**: 在联邦学习中使用单步 Anderson 加速
- **与本文区别**: FedOSAA 是同步设定，本文聚焦异步场景
- **参考价值**: AA 在联邦学习中的实现经验

#### Liu & Wu (2026) - 时变网络上的分布式约束优化 AA
- **核心思想**: 时变网络上的 AA 加速
- **与本文区别**: 聚焦网络拓扑变化，本文聚焦异步时延和客户端异构性
- **参考价值**: 分布式 AA 的理论分析方法

## 研究空白与机会

### 本研究的独特贡献
1. **异步历史池视角**: 不是直接对梯度做 AA，而是利用异步系统自然产生的历史更新池
2. **双场景设计**: 分别处理同构和异构两种情形
3. **统一 safeguard 机制**: trigger + restart + accept/reject 的系统设计

### 与现有工作的核心区别
| 方面 | 现有工作 | 本研究 |
|------|---------|---------|
| 同步/异步 | 大多数是同步 | 聚焦异步场景 |
| AA 对象 | 梯度或残差 | 异步历史更新池 |
| 异构处理 | 通常不考虑 | 专门设计 correction + normalization |
| Safeguard | 简单版本 | 系统性设计 |

## 下一步行动

### Phase 2: Survey 阶段计划
1. 扩展搜索范围到 2020-2025 年
2. 收集 35-80 篇高质量论文
3. 按主题分类：AA 基础、异步 FL、异构 FL、AA+分布式优化
4. 深度阅读 8-15 篇核心论文

### 优先阅读列表
1. **FedOSAA** (2025) - 最相关工作
2. **Walker & Ni (2011)** - AA 经典基础
3. **Feng et al. (2024)** - AA 收敛分析
4. **FedBuff** (2022) - 异步聚合基准
5. **SCAFFOLD** (2020) - Control variates 基准
6. **FedProx** (2020) - Proximal 正则化基准
7. **Wei et al. (2021)** - 随机 AA 基础

## 文献资源链接

### GitHub 资源
- [awesome-asynchronous-federated-learning](https://github.com/beiyuouo/awesome-asynchronous-federated-learning) - 异步联邦学习论文汇总

### 关键论文链接
- FedBuff: https://proceedings.mlr.press/v151/nguyen22b/nguyen22b.pdf
- FedAsync: https://arxiv.org/abs/1903.03934
- SCAFFOLD: https://proceedings.mlr.press/v119/karimireddy20a/karimireddy20a.pdf
- FedProx: https://proceedings.mlsys.org/paper_files/paper/2020/file/1f5fe83998a09396ebe6477d9475ba0c-Paper.pdf
- FedNova: https://proceedings.neurips.cc/paper_files/paper/2020/file/564127c03caab942e503ee6f810f54fd-Paper.pdf

---

*✅ Phase 1 complete. Output: frontier.md. Ready to proceed to Phase 2: Survey.*
