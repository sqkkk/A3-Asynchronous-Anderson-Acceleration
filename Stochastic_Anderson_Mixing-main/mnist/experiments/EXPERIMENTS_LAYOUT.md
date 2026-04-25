# Experiments Layout

`experiments/` 根目录已经清理为“目录优先”的结构。

## Principle

- 现有的 `*_fig` / `*_logs` / `*_watch` 目录保持原位，不额外改动
- 原来平铺在 `experiments/` 根目录下的大量结果文件，统一移入 `experiments/by_family/`
- 这样做的目标是减少根目录噪声，同时尽量不破坏已有图表和日志目录的路径习惯

## Main Entry

- `experiments/by_family/`

## Current Structure

- `by_family/cifar100/resnet18/main_e300_ra_re_ls/`
  - CIFAR-100 / ResNet18 / 300e / RandAug+RE+LS 四设置主线结果文件
- `by_family/cifar100/resnet18/tuning/`
  - CIFAR-100 / ResNet18 / AsyncSAM+RMS 调参结果
- `by_family/cifar100/resnet18/pilots/`
  - CIFAR-100 / ResNet18 的 highbudget / strongaug / focused80 / watch 类 pilot
- `by_family/cifar100/resnet32/`
  - CIFAR-100 / ResNet32 相关结果
- `by_family/cifar100/resnext29/`
  - CIFAR-100 / ResNeXt29 相关结果
- `by_family/cifar100/other_models/`
  - CIFAR-100 上其它 backbone，例如 `r56 / r110`
- `by_family/cifar100/misc/`
  - CIFAR-100 其它零散结果
- `by_family/cifar10/resnet32/`
  - CIFAR-10 / ResNet32 相关结果
- `by_family/cifar10/resnet56/`
  - CIFAR-10 / ResNet56 相关结果
- `by_family/cifar10/workers20/`
  - CIFAR-10 / 20 workers 相关结果
- `by_family/cifar10/legacy_and_misc/`
  - CIFAR-10 旧版主线、legacy92、debug、fedac 等零散结果
- `by_family/cifar10/misc/`
  - CIFAR-10 其它暂未继续细分的文件
- `by_family/mnist_afl_legacy/`
  - MNIST、AFL、早期异步分布式、FedAC smoke 等 legacy 结果
- `by_family/validation_misc/`
  - validation / validate 前缀的结果
- `by_family/other/`
  - 目前仅保留少量无法自然归类的边角日志

## Notes

- 如果以后新实验还是继续往 `experiments/` 根目录直接落文件，建议优先按上面的结构手动归档到 `by_family/`
- 如果某张图或某个脚本还引用旧的根目录文件路径，需要相应改成 `by_family/...` 下的新路径
